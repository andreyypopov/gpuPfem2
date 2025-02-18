#ifndef NUMERICAL_INTEGRATOR_3D_CUH
#define NUMERICAL_INTEGRATOR_3D_CUH

#include "../common/constants.h"
#include "../common/matrix4x4.cuh"
#include "../mesh_3d.cuh"
#include "../linear_algebra/sparse_matrix.cuh"

__device__ inline Point3 shapeFuncGrad3D(int i) {
    switch (i)
    {
    case 0:
        return { 1.0, 0.0, 0.0 };
    case 1:
        return { 0.0, 1.0, 0.0 };
    case 2:
        return { 0.0, 0.0, 1.0 };
    case 3:
        return { -1.0, -1.0, -1.0 };
    default:
        return Point3();
    }
}

__device__ inline void addLocalToGlobal3D(const uint4& cell, const double volume, const Matrix4x4& localMatrix, const Vector4& localRhs,
    const int* rowOffset, const int* colIndices, double* matrixValues, double* rhsVector)
{
    for (int i = 0; i < 4; ++i) {
        const unsigned int vertexIndexI = *(&cell.x + i);

        //elements of the global matrix
        const int indexOfFirstElementInRow = rowOffset[vertexIndexI];

        int numElementsInRow = rowOffset[vertexIndexI + 1] - indexOfFirstElementInRow;
        for (int j = 0; j < 4; ++j) {
            const unsigned int vertexIndexJ = *(&cell.x + j);
            const int index = indexBinarySearch(vertexIndexJ, &colIndices[indexOfFirstElementInRow], numElementsInRow);

            if (index >= 0)
                atomicAdd(&matrixValues[index + indexOfFirstElementInRow], localMatrix(i, j) * volume);
        }

        //element of the right hand side vector
        atomicAdd(&rhsVector[vertexIndexI], localRhs(i) * volume);
    }
}

class NumericalIntegrator3D
{
public:
    NumericalIntegrator3D(const Mesh3D &mesh_)
        : mesh(mesh_)
    {};

    virtual ~NumericalIntegrator3D(){ };

protected:
    const Mesh3D &mesh;

};

#endif // NUMERICAL_INTEGRATOR_2D_CUH
