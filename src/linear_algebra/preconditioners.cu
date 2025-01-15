#include "preconditioners.cuh"

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

Preconditioner::Preconditioner(int n_, const LinearAlgebra *LA_)
    : n(n_)
    , LA(LA_)
{
    gpuBlocks = blocksForSize(n);
}

PreconditionerJacobi::PreconditionerJacobi(int n_, const LinearAlgebra *LA_)
    : Preconditioner(n_, LA_)
{
    invDiagValues.allocate(n_);
}

void PreconditionerJacobi::initialize(const SparseMatrixCSR &csrMatrix)
{
    extractDiagonal<<<gpuBlocks, gpuThreads>>>(n, invDiagValues.data, csrMatrix.getRowOffset(), csrMatrix.getColIndices(), csrMatrix.getMatrixValues());
}

void PreconditionerJacobi::applyPreconditioner(double *dest, const double *src)
{
    applyJacobiPreconditioner<<<gpuBlocks, gpuThreads>>>(n, dest, src, invDiagValues.data);
}

PreconditionerILU::PreconditionerILU(const SparseMatrixCSR &matrix, const LinearAlgebra *LA_)
    : Preconditioner(matrix.getRows(), LA_)
    , nnz(matrix.getTotalElements())
    , alpha(1.0)
{
    checkCusparseErrors(cusparseCreateCsrilu02Info(&iluInfo));

    checkCusparseErrors(cusparseSpSV_createDescr(&lSpsvDescription));
    checkCusparseErrors(cusparseSpSV_createDescr(&uSpsvDescription));

    checkCusparseErrors(cusparseCreateMatDescr(&iluMatrix));
    checkCusparseErrors(cusparseSetMatIndexBase(iluMatrix, CUSPARSE_INDEX_BASE_ZERO));
    checkCusparseErrors(cusparseSetMatType(iluMatrix, CUSPARSE_MATRIX_TYPE_GENERAL));

    matrixValues.allocate(nnz);

    checkCusparseErrors(cusparseCreateCsr(&lMatrix, n, n, nnz,
        matrix.getRowOffset(), matrix.getColIndices(), matrixValues.data,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    checkCusparseErrors(cusparseCreateCsr(&uMatrix, n, n, nnz,
        matrix.getRowOffset(), matrix.getColIndices(), matrixValues.data,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseFillMode_t lFillMode = CUSPARSE_FILL_MODE_LOWER;
    cusparseFillMode_t uFillMode = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t lDiagType = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseDiagType_t uDiagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
    checkCusparseErrors(cusparseSpMatSetAttribute(lMatrix, CUSPARSE_SPMAT_FILL_MODE, &lFillMode, sizeof(lFillMode)));
    checkCusparseErrors(cusparseSpMatSetAttribute(uMatrix, CUSPARSE_SPMAT_FILL_MODE, &uFillMode, sizeof(uFillMode)));
    checkCusparseErrors(cusparseSpMatSetAttribute(lMatrix, CUSPARSE_SPMAT_DIAG_TYPE, &lDiagType, sizeof(lDiagType)));
    checkCusparseErrors(cusparseSpMatSetAttribute(uMatrix, CUSPARSE_SPMAT_DIAG_TYPE, &uDiagType, sizeof(uDiagType)));

    auxVector.allocate(n);
    //at this point all vector description are set to the auxiliary vector
    //(will be later updated for src and dest vectors at the stage of preconditioner initialization)
    checkCusparseErrors(cusparseCreateDnVec(&auxVec, n, auxVector.data, CUDA_R_64F));
    checkCusparseErrors(cusparseCreateDnVec(&destVec, n, auxVector.data, CUDA_R_64F));
    checkCusparseErrors(cusparseCreateDnVec(&srcVec, n, auxVector.data, CUDA_R_64F));

    const int iluBufferSize = LA->incompleteLU_bufferSize(iluMatrix, matrix.getRowOffset(), matrix.getColIndices(), matrix.getMatrixValues(), iluInfo, n, nnz);
    const int lBuffersize = LA->solveSparseTriangularSystem_bufferSize(lMatrix, srcVec, auxVec, &alpha, lSpsvDescription);
    const int uBufferSize = LA->solveSparseTriangularSystem_bufferSize(uMatrix, auxVec, destVec, &alpha, uSpsvDescription);

    checkCudaErrors(cudaMalloc(&iluBuffer, iluBufferSize));
    checkCudaErrors(cudaMalloc(&lSpsvBuffer, lBuffersize));
    checkCudaErrors(cudaMalloc(&uSpsvBuffer, uBufferSize));
}

PreconditionerILU::~PreconditionerILU()
{
    checkCudaErrors(cudaFree(iluBuffer));
    checkCudaErrors(cudaFree(lSpsvBuffer));
    checkCudaErrors(cudaFree(uSpsvBuffer));
    checkCusparseErrors(cusparseDestroyCsrilu02Info(iluInfo));
    checkCusparseErrors(cusparseSpSV_destroyDescr(lSpsvDescription));
    checkCusparseErrors(cusparseSpSV_destroyDescr(uSpsvDescription));

    checkCusparseErrors(cusparseDestroyMatDescr(iluMatrix));
    checkCusparseErrors(cusparseDestroySpMat(lMatrix));
    checkCusparseErrors(cusparseDestroySpMat(uMatrix));
    checkCusparseErrors(cusparseDestroyDnVec(auxVec));
    checkCusparseErrors(cusparseDestroyDnVec(destVec));
    checkCusparseErrors(cusparseDestroyDnVec(srcVec));
}

void PreconditionerILU::initialize(const SparseMatrixCSR &csrMatrix)
{
    copy_d2d(csrMatrix.getMatrixValues(), matrixValues.data, nnz);

    LA->incompleteLU(iluMatrix, csrMatrix.getRowOffset(), csrMatrix.getColIndices(), matrixValues.data, iluInfo, n, nnz, iluBuffer);
    analysisRequired = true;
}

void PreconditionerILU::applyPreconditioner(double *dest, const double *src)
{
    if(analysisRequired){
        checkCusparseErrors(cusparseDnVecSetValues(srcVec, (void*)src));
        checkCusparseErrors(cusparseDnVecSetValues(destVec, (void*)dest));

        LA->solveSparseTriangularSystem_analysis(lMatrix, srcVec, auxVec, &alpha, lSpsvDescription, lSpsvBuffer);
        LA->solveSparseTriangularSystem_analysis(uMatrix, auxVec, destVec, &alpha, uSpsvDescription, uSpsvBuffer);
    }

    LA->solveSparseTriangularSystem(lMatrix, srcVec, auxVec, &alpha, lSpsvDescription);
    LA->solveSparseTriangularSystem(uMatrix, auxVec, destVec, &alpha, uSpsvDescription);
}

PreconditionerIC::PreconditionerIC(const SparseMatrixCSR &matrix, const LinearAlgebra *LA_)
    : Preconditioner(matrix.getRows(), LA_)
    , nnz(matrix.getTotalElements())
    , alpha(1.0)
{
    checkCusparseErrors(cusparseCreateCsric02Info(&icInfo));

    checkCusparseErrors(cusparseSpSV_createDescr(&lSpsvDescription));
    checkCusparseErrors(cusparseSpSV_createDescr(&ltSpsvDescription));

    checkCusparseErrors(cusparseCreateMatDescr(&icMatrix));
    checkCusparseErrors(cusparseSetMatIndexBase(icMatrix, CUSPARSE_INDEX_BASE_ZERO));
    checkCusparseErrors(cusparseSetMatType(icMatrix, CUSPARSE_MATRIX_TYPE_GENERAL));

    matrixValues.allocate(nnz);

    checkCusparseErrors(cusparseCreateCsr(&lMatrix, n, n, nnz,
        matrix.getRowOffset(), matrix.getColIndices(), matrixValues.data,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseFillMode_t lFillMode = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t lDiagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
    checkCusparseErrors(cusparseSpMatSetAttribute(lMatrix, CUSPARSE_SPMAT_FILL_MODE, &lFillMode, sizeof(lFillMode)));
    checkCusparseErrors(cusparseSpMatSetAttribute(lMatrix, CUSPARSE_SPMAT_DIAG_TYPE, &lDiagType, sizeof(lDiagType)));

    auxVector.allocate(n);
    //at this point all vector description are set to the auxiliary vector
    //(will be later updated for src and dest vectors at the stage of preconditioner initialization)
    checkCusparseErrors(cusparseCreateDnVec(&auxVec, n, auxVector.data, CUDA_R_64F));
    checkCusparseErrors(cusparseCreateDnVec(&destVec, n, auxVector.data, CUDA_R_64F));
    checkCusparseErrors(cusparseCreateDnVec(&srcVec, n, auxVector.data, CUDA_R_64F));

    const int icBufferSize = LA->incompleteCholesky_bufferSize(icMatrix, matrix.getRowOffset(), matrix.getColIndices(), matrix.getMatrixValues(), icInfo, n, nnz);
    const int lBuffersize = LA->solveSparseTriangularSystem_bufferSize(lMatrix, srcVec, auxVec, &alpha, lSpsvDescription);
    const int ltBufferSize = LA->solveSparseTriangularSystem_bufferSize(lMatrix, auxVec, destVec, &alpha, ltSpsvDescription, true);

    checkCudaErrors(cudaMalloc(&icBuffer, icBufferSize));
    checkCudaErrors(cudaMalloc(&lSpsvBuffer, lBuffersize));
    checkCudaErrors(cudaMalloc(&ltSpsvBuffer, ltBufferSize));
}

PreconditionerIC::~PreconditionerIC()
{
    checkCudaErrors(cudaFree(icBuffer));
    checkCudaErrors(cudaFree(lSpsvBuffer));
    checkCudaErrors(cudaFree(ltSpsvBuffer));
    checkCusparseErrors(cusparseDestroyCsric02Info(icInfo));
    checkCusparseErrors(cusparseSpSV_destroyDescr(lSpsvDescription));
    checkCusparseErrors(cusparseSpSV_destroyDescr(ltSpsvDescription));

    checkCusparseErrors(cusparseDestroyMatDescr(icMatrix));
    checkCusparseErrors(cusparseDestroySpMat(lMatrix));
    checkCusparseErrors(cusparseDestroyDnVec(auxVec));
    checkCusparseErrors(cusparseDestroyDnVec(destVec));
    checkCusparseErrors(cusparseDestroyDnVec(srcVec));
}

void PreconditionerIC::initialize(const SparseMatrixCSR &csrMatrix)
{
    copy_d2d(csrMatrix.getMatrixValues(), matrixValues.data, nnz);

    LA->incompleteCholesky(icMatrix, csrMatrix.getRowOffset(), csrMatrix.getColIndices(), matrixValues.data, icInfo, n, nnz, icBuffer);
    analysisRequired = true;
}

void PreconditionerIC::applyPreconditioner(double *dest, const double *src)
{
    if(analysisRequired){
        checkCusparseErrors(cusparseDnVecSetValues(srcVec, (void*)src));
        checkCusparseErrors(cusparseDnVecSetValues(destVec, (void*)dest));

        LA->solveSparseTriangularSystem_analysis(lMatrix, srcVec, auxVec, &alpha, lSpsvDescription, lSpsvBuffer);
        LA->solveSparseTriangularSystem_analysis(lMatrix, auxVec, destVec, &alpha, ltSpsvDescription, ltSpsvBuffer, true);
    }

    LA->solveSparseTriangularSystem(lMatrix, srcVec, auxVec, &alpha, lSpsvDescription);
    LA->solveSparseTriangularSystem(lMatrix, auxVec, destVec, &alpha, ltSpsvDescription, true);
}
