#include "Dirichlet_bcs.cuh"

__global__ void kEnforceValueForBoundaryDoFs(int n, const DirichletNode* boundaryValues,
    const int* rowOffset, const int* colIndices, double* matrixValues, double* rhsVector)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const unsigned int dofIndex = boundaryValues[idx].nodeIdx;

        const int start = rowOffset[dofIndex];
        const int end = rowOffset[dofIndex + 1];
        for (int i = start; i < end; ++i)
            matrixValues[i] = (colIndices[i] == dofIndex) ? 1.0 : 0.0;

        rhsVector[dofIndex] = boundaryValues[idx].bcValue;
    }
}

__global__ void kAccountForBoundaryValues(int n, const DirichletNode* boundaryValues,
    const int* rowOffset, const int* colIndices, double* matrixValues, double* rhsVector)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const unsigned int dofIndex = boundaryValues[idx].nodeIdx;
        const double minusBoundaryVal = -boundaryValues[idx].bcValue;

        const int start = rowOffset[dofIndex];
        const int end = rowOffset[dofIndex + 1];

        //iterate over dofIndex-th row = iterate over dofIndex-th column
        for (int i = start; i < end; ++i)
            if(colIndices[i] != dofIndex) {
                const int row = colIndices[i];

                //check whether it is not a row for another boundary DoF
                const int start2 = rowOffset[row];
                const int end2 = rowOffset[row + 1];

                const int firstOffDiagonalElementIdx = start2 + (colIndices[start2] == row);
                if (fabs(matrixValues[firstOffDiagonalElementIdx]) < CONSTANTS::DOUBLE_MIN)
                    continue;

                //find index of the element corresponding to DoF in this row
                int idx2 = indexBinarySearch(dofIndex, colIndices + start2, end2 - start2);
                if (idx2 >= 0) {
                    atomicAdd(&rhsVector[row], matrixValues[start2 + idx2] * minusBoundaryVal);
                    matrixValues[start2 + idx2] = 0.0;
                }
            }
    }
}

void DirichletBCs::setupDirichletBCs(const std::vector<DirichletNode> &hostBcs)
{
    DirichletValues.allocate(hostBcs.size());
    copy_h2d(hostBcs.data(), DirichletValues.data, hostBcs.size());

    printf("Number of boundary nodes: %zu\n", hostBcs.size());
}

void DirichletBCs::applyBCs(SparseMatrixCSR& matrix, deviceVector<double>& rhs)
{
    //1. For the rows of the matrix that correspond to boundary DoFs set diagonal element = 1, off-diagonal elements = 0
    //and right hand side element = BCvalue
    unsigned int blocks = blocksForSize(DirichletValues.size);
    kEnforceValueForBoundaryDoFs<<<blocks, gpuThreads>>>(DirichletValues.size, DirichletValues.data, matrix.getRowOffset(),
        matrix.getColIndices(), matrix.getMatrixValues(), rhs.data);

    //2. For other rows subtract aij*bcValue from the right hand side element and set aij = 0
    kAccountForBoundaryValues<<<blocks, gpuThreads>>>(DirichletValues.size, DirichletValues.data, matrix.getRowOffset(),
        matrix.getColIndices(), matrix.getMatrixValues(), rhs.data);

    cudaDeviceSynchronize();
}
