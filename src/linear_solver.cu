#include "linear_solver.cuh"

#include "common/cuda_helper.cuh"
#include "common/cuda_memory.cuh"

__global__ void subtractVectors(int n, double* res, double* v1, double* v2)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
        return;

    res[row] = v1[row] - v2[row];
}

__global__ void updateXR(int n, double *x, double *r, const double *p, const double *Ap, const double *numerator, const double *denominator){
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
        return;

    const double alpha = *numerator / *denominator;

    x[row] += alpha * p[row];
    r[row] -= alpha * Ap[row];
}

__global__ void updateP(int n, double *p, const double *v, const double *numerator, const double *denominator){
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
        return;

    const double beta = *numerator / *denominator;

    p[row] = v[row] + beta * p[row];
}

__global__ void updateVectors(int n, double *p, double *s, double *x, double *r, const double *u, const double *w, const double *alpha, const double *beta)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
        return;

    p[row] = u[row] + (*beta) * p[row];
    s[row] = w[row] + (*beta) * s[row];    
    x[row] += (*alpha) * p[row];
    r[row] -= (*alpha) * s[row];
}

__global__ void updateCoefficients(double *alpha, double *beta, const double *gamma_new, const double *gamma_old, const double *delta)
{
    *beta = (*gamma_new) / (*gamma_old);
    *alpha = 1.0 / ((*delta) / (*gamma_new) - *beta / (*alpha));
}

__global__ void initCoefficients(double *alpha, double *beta, const double *gamma, const double *delta)
{
    *alpha = (*gamma) / (*delta);
    *beta = 0;
}

__global__ void extractDiagonal(int n, double *invDiagonal, const int *rowPtr, const int *colIndex, const double *matrixVal)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
        return;

    const int startElem = rowPtr[row];
    const int endElem = rowPtr[row + 1];
    int diagIndex = indexBinarySearch(row, colIndex + startElem, endElem - startElem);
    if(diagIndex >= 0)
        invDiagonal[row] = 1.0 / matrixVal[startElem + diagIndex];
}

__global__ void applyJacobiPreconditioner(int n, double *dest, const double *src, const double *preconditioner)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
        return;

    dest[row] = src[row] * preconditioner[row];
}

SolverCG::SolverCG(double tolerance, int max_iterations)
    : aSpmv(1.0)
    , bSpmv(0.0)
    , tolerance(tolerance)
    , tolerance_squared(tolerance * tolerance)
    , maxIterations(max_iterations)
{
    checkCublasErrors(cublasCreate(&cublasHandle));
    checkCusparseErrors(cusparseCreate(&cusparseHandle));
}

SolverCG::~SolverCG(){
    checkCublasErrors(cublasDestroy(cublasHandle));
    checkCusparseErrors(cusparseDestroySpMat(matA));
    checkCusparseErrors(cusparseDestroyDnVec(vecX));
    checkCusparseErrors(cusparseDestroyDnVec(vecY));
    checkCusparseErrors(cusparseDestroy(cusparseHandle));
    checkCudaErrors(cudaFree(dBuffer));

    if (ChronopolousGear) {
        free_device(alpha_k);
        free_device(beta_k);
        free_device(delta_k);
    } else
        free_device(pkApk);

    free_device(gamma_kp);
    free_device(gamma_k);
}

void SolverCG::init(const SparseMatrixCSR &matrix, bool usePreconditioning){
    this->n = matrix.getRows();
    this->nnz = matrix.getTotalElements();
    this->usePreconditioning = usePreconditioning;

    gpuBlocks = blocksForSize(n);

    rk.allocate(n);
    if(usePreconditioning){
        if(ChronopolousGear)
            uk.allocate(n);
        else
            zk.allocate(n);

        invDiagValues.allocate(n);        
    }
    
    if (ChronopolousGear) {
        wk.allocate(n);
        sk.allocate(n);
    } else
        Apk.allocate(n);
    
    pk.allocate(n);
       
    if (ChronopolousGear) {
        allocate_device(&alpha_k, 1);
        allocate_device(&beta_k, 1);
        allocate_device(&delta_k, 1);
    } else
        allocate_device(&pkApk, 1);

    allocate_device(&gamma_kp, 1);
    allocate_device(&gamma_k, 1);

    size_t bufferSize = 0;

    checkCusparseErrors(cusparseCreateCsr(&matA, n, n, nnz,
        matrix.getRowOffset(), matrix.getColIndices(), matrix.getMatrixValues(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    if (ChronopolousGear) {
        double *v = usePreconditioning ? uk.data : rk.data;
        checkCusparseErrors(cusparseCreateDnVec(&vecX, n, v, CUDA_R_64F));
        checkCusparseErrors(cusparseCreateDnVec(&vecY, n, wk.data, CUDA_R_64F));
    }
    else {
        checkCusparseErrors(cusparseCreateDnVec(&vecX, n, pk.data, CUDA_R_64F));
        checkCusparseErrors(cusparseCreateDnVec(&vecY, n, Apk.data, CUDA_R_64F));
    }
    checkCusparseErrors(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));
}

bool SolverCG::solveChronopolousGear(const SparseMatrixCSR &A, deviceVector<double> &x, const deviceVector<double> &b)
{
    bool converged = false;

    if(usePreconditioning)
        extractDiagonal<<<gpuBlocks, gpuThreads>>>(n, invDiagValues.data, A.getRowOffset(), A.getColIndices(), A.getMatrixValues());

    //x0 = 0
    zero_value_device(x.data, n);
    //r0 = b - A*x0 = b;
    copy_d2d(b.data, rk.data, n);

    //u_i = M^(-1)*r_i
    if(usePreconditioning)
        applyJacobiPreconditioner<<<gpuBlocks, gpuThreads>>>(n, uk.data, rk.data, invDiagValues.data);

    //w0 = A*u0
    checkCusparseErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    
    double *v = usePreconditioning ? uk.data : rk.data;
    //gamma0 = (r0, u0)
    checkCublasErrors(cublasDdot(cublasHandle, n, rk.data, 1, v, 1, gamma_kp));
    //delta0 = (w0, u0)
    checkCublasErrors(cublasDdot(cublasHandle, n, wk.data, 1, v, 1, delta_k));
    //alpha0 = delta0 / gamma0; beta0 = 0
    initCoefficients<<<1, 1>>>(alpha_k, beta_k, gamma_kp, delta_k);

    int it = 0;
    while(it < maxIterations){
        ++it;

        updateVectors<<<gpuBlocks, gpuThreads>>>(n, pk.data, sk.data, x.data, rk.data, v, wk.data, alpha_k, beta_k);

        //u_i = M^(-1)*r_i
        if(usePreconditioning)
            applyJacobiPreconditioner<<<gpuBlocks, gpuThreads>>>(n, uk.data, rk.data, invDiagValues.data);

        //w_i = A*u_i
        checkCusparseErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        
        std::swap(gamma_kp, gamma_k);
        //gamma_i = (r_i, u_i)
        checkCublasErrors(cublasDdot(cublasHandle, n, rk.data, 1, v, 1, gamma_kp));

        copy_d2h(gamma_kp, &residual_norm, 1);
        if(residual_norm < tolerance_squared){
            converged = true;
            break;
        }

        //delta_i = (w_i, u_i)
        checkCublasErrors(cublasDdot(cublasHandle, n, wk.data, 1, v, 1, delta_k));
        //calculate alpha_i, beta_i
        updateCoefficients<<<1, 1>>>(alpha_k, beta_k, gamma_kp, gamma_k, delta_k);
    }

    if(converged)
        printf("Solver converged with residual=%e, no. of iterations=%d\n", std::sqrt(residual_norm), it);
    else
        printf("Solver failed to converge\n");

    return converged;
}

bool SolverCG::solve(const SparseMatrixCSR &A, deviceVector<double> &x, const deviceVector<double> &b){
    bool converged = false;

    if(usePreconditioning)
        extractDiagonal<<<gpuBlocks, gpuThreads>>>(n, invDiagValues.data, A.getRowOffset(), A.getColIndices(), A.getMatrixValues());

    //x0 = 0
    zero_value_device(x.data, n);
    //r0 = b - A*x0 = b;
    copy_d2d(b.data, rk.data, n);

    //z_0 = M^(-1)*r_0
    if(usePreconditioning)
        applyJacobiPreconditioner<<<gpuBlocks, gpuThreads>>>(n, zk.data, rk.data, invDiagValues.data);

    double *v = usePreconditioning ? zk.data : rk.data;
    copy_d2d(v, pk.data, n);

    //(r_0, z_0)
    checkCublasErrors(cublasDdot(cublasHandle, n, rk.data, 1, v, 1, gamma_kp));

    int it = 0;
    while(it < maxIterations){
        ++it;

        checkCusparseErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        //(p_i, Ap_i)
        checkCublasErrors(cublasDdot(cublasHandle, n, pk.data, 1, Apk.data, 1, pkApk));

        //update x_i and r_i vectors
        updateXR<<<gpuBlocks, gpuThreads>>>(n, x.data, rk.data, pk.data, Apk.data, gamma_kp, pkApk);

        //z_i = M^(-1)*r_i
        if(usePreconditioning)
            applyJacobiPreconditioner<<<gpuBlocks, gpuThreads>>>(n, zk.data, rk.data, invDiagValues.data);

        std::swap(gamma_kp, gamma_k);
        //gamma_i = (r_i, z_i)
        checkCublasErrors(cublasDdot(cublasHandle, n, rk.data, 1, v, 1, gamma_kp));
        copy_d2h(gamma_kp, &residual_norm, 1);
        if (residual_norm < tolerance_squared) {
            converged = true;
            break;
        }

        //update the p_i vector
        updateP<<<gpuBlocks, gpuThreads>>>(n, pk.data, v, gamma_kp, gamma_k);
    }
    
    if(converged)
        printf("Solver converged with residual=%e, no. of iterations=%d\n", std::sqrt(residual_norm), it);
    else
        printf("Solver failed to converge\n");

    return converged;
}
