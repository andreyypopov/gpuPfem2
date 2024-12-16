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

PreconditionerJacobi::PreconditionerJacobi(int n_)
    : Preconditioner(n_)
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

Preconditioner::Preconditioner(int n_)
    : n(n_)
{
    gpuBlocks = blocksForSize(n);
}
