#include "linear_solver.cuh"

#include "common/cuda_helper.cuh"
#include "common/cuda_math.cuh"
#include "common/cuda_memory.cuh"

__global__ void subtractVectors(int n, double* res, double* v1, double* v2)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
        return;

    res[row] = v1[row] - v2[row];
}

__global__ void scaleVector(int n, double *out, const double *in, const double *factor, bool divide = false)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
        return;

    if(divide)
        out[row] = in[row] / *factor;
    else
        out[row] = in[row] * *factor;
}

__global__ void daxpy(int n, double *out, const double *in, const double *factor, bool changeSign = false)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
        return;

    if(changeSign)
        out[row] -= *factor * in[row];
    else
        out[row] += *factor * in[row];
}

__global__ void updateHcolumn(int k, double *h_k, double *h_kp, double *cs, double *sn, double *beta)
{
    //the kernel should be run precisely by 1 core as the work is performed in a strictly serial way 
    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;

    //1. Update h_1k,...h_kk
    double tmp;
    for(int i = 0; i < k; ++i){
        tmp = cs[i] * h_k[i] + sn[i] * h_k[i + 1];
        h_k[i + 1] = -sn[i] * h_k[i] + cs[i] * h_k[i + 1];
        h_k[i] = tmp;
    }

    //2. Get cs_k, sn_k for h_kk and h_{k+1}k
    Point2 cssn_k = GivensRotation(h_k[k], *h_kp);
    cs[k] = cssn_k.x;
    sn[k] = cssn_k.y;

    //3. Update h_kk and h_{k+1}k
    h_k[k] = cs[k] * h_k[k] + sn[k] * *h_kp;
    *h_kp = 0.0;

    //4. Update beta_k and beta_{k+1}
    beta[k + 1] = -sn[k] * beta[k];
    beta[k] *= cs[k];    
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

LinearSolver::LinearSolver(double tolerance, int max_iterations)
    : aSpmv(1.0)
    , bSpmv(0.0)
    , tolerance(tolerance)
    , tolerance_squared(tolerance * tolerance)
    , maxIterations(max_iterations)
{
    checkCublasErrors(cublasCreate(&cublasHandle));
    //needed for correct execution of functions which return scalar result (dot, nrm2)
    checkCublasErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE));

    checkCusparseErrors(cusparseCreate(&cusparseHandle));
}

LinearSolver::~LinearSolver()
{
    checkCublasErrors(cublasDestroy(cublasHandle));
    checkCusparseErrors(cusparseDestroySpMat(matA));
    checkCusparseErrors(cusparseDestroyDnVec(vecX));
    checkCusparseErrors(cusparseDestroyDnVec(vecY));
    checkCusparseErrors(cusparseDestroy(cusparseHandle));
    checkCudaErrors(cudaFree(dBuffer));
}

void LinearSolver::init(const SparseMatrixCSR& matrix, bool usePreconditioning) {
    this->n = matrix.getRows();
    this->nnz = matrix.getTotalElements();
    this->usePreconditioning = usePreconditioning;

    gpuBlocks = blocksForSize(n);

    if(usePreconditioning)
        invDiagValues.allocate(n);
    
    checkCusparseErrors(cusparseCreateCsr(&matA, n, n, nnz,
        matrix.getRowOffset(), matrix.getColIndices(), matrix.getMatrixValues(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    csrMatrix = &matrix;
}

bool LinearSolver::solve(const SparseMatrixCSR &A, deviceVector<double> &x, const deviceVector<double> &b)
{
    if(csrMatrix != &A){
        csrMatrix = &A;
        checkCusparseErrors(cusparseCsrSetPointers(matA, A.getRowOffset(), A.getColIndices(), A.getMatrixValues()));
    }

    if(usePreconditioning)
        extractDiagonal<<<gpuBlocks, gpuThreads>>>(n, invDiagValues.data, A.getRowOffset(), A.getColIndices(), A.getMatrixValues());

    return true;
}

SolverCG::SolverCG(double tolerance, int max_iterations)
    : LinearSolver(tolerance, max_iterations)
{
}

SolverCG::~SolverCG(){
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
    LinearSolver::init(matrix, usePreconditioning);

    rk.allocate(n);
    if(usePreconditioning){
        if(ChronopolousGear)
            uk.allocate(n);
        else
            zk.allocate(n);
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

    if (ChronopolousGear) {
        double *v = usePreconditioning ? uk.data : rk.data;
        checkCusparseErrors(cusparseCreateDnVec(&vecX, n, v, CUDA_R_64F));
        checkCusparseErrors(cusparseCreateDnVec(&vecY, n, wk.data, CUDA_R_64F));
    }
    else {
        checkCusparseErrors(cusparseCreateDnVec(&vecX, n, pk.data, CUDA_R_64F));
        checkCusparseErrors(cusparseCreateDnVec(&vecY, n, Apk.data, CUDA_R_64F));
    }

    size_t bufferSize = 0;

    checkCusparseErrors(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG1, &bufferSize));
    checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));

    checkCusparseErrors(cusparseSpMV_preprocess(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG1, dBuffer));
}

bool SolverCG::solveChronopolousGear(const SparseMatrixCSR &A, deviceVector<double> &x, const deviceVector<double> &b)
{
    bool converged = false;

    if(usePreconditioning)
        extractDiagonal<<<gpuBlocks, gpuThreads>>>(n, invDiagValues.data, A.getRowOffset(), A.getColIndices(), A.getMatrixValues());

    //save main pointer for the vector used in SpMV
    void* vecPointer = nullptr;
    checkCusparseErrors(cusparseDnVecGetValues(vecX, &vecPointer));

    //temporarily set the vector in SpMV to x0
    checkCusparseErrors(cusparseDnVecSetValues(vecX, x.data));

    //calculate A*x0
    checkCusparseErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_CSR_ALG1, dBuffer));

    //r0 = b - A*x0
    subtractVectors<<<gpuBlocks, gpuThreads>>>(n, rk.data, b.data, wk.data);

    //set pointer of the vector in SpMV back to normal
    checkCusparseErrors(cusparseDnVecSetValues(vecX, vecPointer));

    //u_i = M^(-1)*r_i
    if(usePreconditioning)
        applyJacobiPreconditioner<<<gpuBlocks, gpuThreads>>>(n, uk.data, rk.data, invDiagValues.data);

    //w0 = A*u0
    checkCusparseErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1, dBuffer));
    
    double *v = usePreconditioning ? uk.data : rk.data;
    //gamma0 = (r0, u0)
    checkCublasErrors(cublasDdot(cublasHandle, n, rk.data, 1, v, 1, gamma_kp));

    copy_d2h(gamma_kp, &residual_norm, 1);
    if(residual_norm < tolerance_squared){
        printf("Solver converged with residual=%e, no. of iterations=0\n", std::sqrt(residual_norm));
        return true;
    }

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
                CUSPARSE_SPMV_CSR_ALG1, dBuffer));
        
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

    LinearSolver::solve(A, x, b);

    //save main pointer for the vector used in SpMV
    void* vecPointer = nullptr;
    checkCusparseErrors(cusparseDnVecGetValues(vecX, &vecPointer));

    //temporarily set the vector in SpMV to x0
    checkCusparseErrors(cusparseDnVecSetValues(vecX, x.data));

    //calculate A*x0
    checkCusparseErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_CSR_ALG1, dBuffer));

    //r0 = b - A*x0
    subtractVectors<<<gpuBlocks, gpuThreads>>>(n, rk.data, b.data, Apk.data);

    //set pointer of the vector in SpMV back to normal
    checkCusparseErrors(cusparseDnVecSetValues(vecX, vecPointer));

    //z_0 = M^(-1)*r_0
    if(usePreconditioning)
        applyJacobiPreconditioner<<<gpuBlocks, gpuThreads>>>(n, zk.data, rk.data, invDiagValues.data);

    double *v = usePreconditioning ? zk.data : rk.data;
    copy_d2d(v, pk.data, n);

    //(r_0, z_0)
    checkCublasErrors(cublasDdot(cublasHandle, n, rk.data, 1, v, 1, gamma_kp));

    copy_d2h(gamma_kp, &residual_norm, 1);
    if(residual_norm < tolerance_squared){
        printf("Solver converged with residual=%e, no. of iterations=0\n", std::sqrt(residual_norm));
        return true;
    }

    int it = 0;
    while(it < maxIterations){
        ++it;

        checkCusparseErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_CSR_ALG1, dBuffer));

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

SolverGMRES::SolverGMRES(double tolerance, int max_iterations)
    : LinearSolver(tolerance, max_iterations)
{
    allocate_device(&aux, 1);
    allocate_device(&d_abSpmv, 1);

    const double tmp = 1.0;
    copy_h2d(&tmp, d_abSpmv, 1);
}

SolverGMRES::~SolverGMRES()
{
    free_device(aux);
    free_device(d_abSpmv);
}

void SolverGMRES::init(const SparseMatrixCSR &matrix, bool usePreconditioning)
{
    LinearSolver::init(matrix, usePreconditioning);

    if (maxIterations > matrix.getRows()) {
        printf("Warning: maximum number of iterations %d is greater than the number of rows in the matrix. Setting it to %d\n", maxIterations, matrix.getRows());
        maxIterations = matrix.getRows();
    }

    cs.allocate(maxIterations);
    sn.allocate(maxIterations);
    beta.allocate(maxIterations + 1);
    Hmatrix.allocate(maxIterations * maxIterations);
    Vmatrix.allocate(n * (maxIterations + 1));
    y.allocate(n);

    //at this point exact locations are not important (will be set and updated at each iteration)
    checkCusparseErrors(cusparseCreateDnVec(&vecX, n, Vmatrix.data, CUDA_R_64F));
    checkCusparseErrors(cusparseCreateDnVec(&vecY, n, Vmatrix.data + n, CUDA_R_64F));

    size_t bufferSize = 0;

    checkCusparseErrors(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG1, &bufferSize));
    checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));

    checkCusparseErrors(cusparseSpMV_preprocess(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG1, dBuffer));
}

bool SolverGMRES::solve(const SparseMatrixCSR &A, deviceVector<double> &x, const deviceVector<double> &b)
{
    bool converged = false;

    LinearSolver::solve(A, x, b);
    
    //temporarily set the vector in SpMV to x0
    checkCusparseErrors(cusparseDnVecSetValues(vecX, x.data));

    v_kp = Vmatrix.data;
    checkCusparseErrors(cusparseDnVecSetValues(vecY, v_kp));

    //calculate A*x0
    checkCusparseErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_CSR_ALG1, dBuffer));

    //r0 = b - A*x0
    subtractVectors<<<gpuBlocks, gpuThreads>>>(n, v_kp, b.data, v_kp);

    //r0 := M^(-1) * r0
    if(usePreconditioning)
        applyJacobiPreconditioner<<<gpuBlocks, gpuThreads>>>(n, v_kp, v_kp, invDiagValues.data);

    //compute ||r0||, which becomes first element of the beta vector
    checkCublasErrors(cublasDnrm2(cublasHandle, n, v_kp, 1, beta.data));

    copy_d2h(beta.data, &residual_norm, 1);
    if (residual_norm < tolerance) {
        printf("Solver converged with residual=%e, no. of iterations=0\n", residual_norm);
        return true;
    }

    //V1 = r0 / ||r0|| = b / ||b|| (or with preconditioner applied)
    scaleVector<<<gpuBlocks, gpuThreads>>>(n, v_kp, v_kp, beta.data, true);

    int it = 0;
    while(it < maxIterations){
        ++it;

        //Arnoldi iteration
        v_k = v_kp;
        v_kp = v_k + n;

        //update the pointers to vectors used in SpMV
        checkCusparseErrors(cusparseDnVecSetValues(vecX, v_k));
        checkCusparseErrors(cusparseDnVecSetValues(vecY, v_kp));

        //perform sparse matrix-vector multiplication v_{k+1} = A * v_k using Cusparse
        checkCusparseErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &aSpmv, matA, vecX, &bSpmv, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_CSR_ALG1, dBuffer));

        if(usePreconditioning)
            applyJacobiPreconditioner<<<gpuBlocks, gpuThreads>>>(n, v_kp, v_kp, invDiagValues.data);

        //pointer to the first element in the column to be filled at this iteration
        double *h_1k = Hmatrix.data + maxIterations * (it - 1);

        //Gram-Schmidt orthogonalization
        for(int i = 0; i < it; ++i){
            //h_{ik} = v_{k+1} * v_i
            const double *v_i = Vmatrix.data + n * i;
            double *h_ik = h_1k + i;
            checkCublasErrors(cublasDdot(cublasHandle, n, v_kp, 1, v_i, 1, h_ik));

            daxpy<<<gpuBlocks, gpuThreads>>>(n, v_kp, v_i, h_ik, true);
        }

        //calculate norm ||v_{k+1}||
        checkCublasErrors(cublasDnrm2(cublasHandle, n, v_kp, 1, aux));
        
        //normalize v_{k+1}
        scaleVector<<<gpuBlocks, gpuThreads>>>(n, v_kp, v_kp, aux, true);

        //prepare H matrix for triangular solve
        updateHcolumn<<<1, 1>>>(it - 1, h_1k, aux, cs.data, sn.data, beta.data);

        //last updated value in the beta vector is equal to residual
        copy_d2h(beta.data + it, &residual_norm, 1);
        residual_norm = std::fabs(residual_norm);

        if (residual_norm < tolerance) {
            converged = true;
            break;
        }
    }

    if (converged) {
        //copy beta into y as triangular solve from Cublas overwrites the right hand side vector
        copy_d2d(beta.data, y.data, it);
        checkCublasErrors(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, it, Hmatrix.data,
            maxIterations, y.data, 1));

        //update x = x0 + V * y
        checkCublasErrors(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, it, d_abSpmv, Vmatrix.data, n, y.data, 1, d_abSpmv, x.data, 1));

        printf("Solver converged with residual=%e, no. of iterations=%d\n", residual_norm, it);
    } else
        printf("Solver failed to converge\n");

    return converged;
}
