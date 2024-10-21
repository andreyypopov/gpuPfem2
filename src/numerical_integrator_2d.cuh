#ifndef NUMERICAL_INTEGRATOR_3D_CUH
#define NUMERICAL_INTEGRATOR_3D_CUH

#include "common/constants.h"
#include "mesh_2d.cuh"
#include "sparse_matrix.cuh"

__device__ inline Point2 shapeFuncGrad(int i) {
    switch (i)
    {
    case 0:
        return { 1.0, 0.0 };
    case 1:
        return { 0.0, 1.0 };
    case 2:
        return { -1.0, -1.0 };
    default:
        return Point2();
    }
}

__device__ inline Point2 transformLocalToGlobal(const Point3 &Lcoordinates, const Point2 *triangleVertices){
	return Lcoordinates.x * triangleVertices[0] + Lcoordinates.y * triangleVertices[1] + Lcoordinates.z * triangleVertices[2];
}

__device__ inline Point3 transformGlobalToLocal(const Point2 &globalCoord, const Matrix2x2 &invJacobi, const Point2 &v3){
    Point3 res;
    //invJacobi needs to be transposed here (which means multiplication of its columns by (p - v3))
    const Point2 drv3 = globalCoord - v3;
    res.x = invJacobi(0,0) * drv3.x + invJacobi(1,0) * drv3.y;
    res.y = invJacobi(0,1) * drv3.x + invJacobi(1,1) * drv3.y;
    res.z = 1.0 - res.x - res.y;

    return res;
}

__device__ inline void addLocalToGlobal(const uint3& triangle, const double area, const Matrix3x3& localMatrix, const Vector3& localRhs,
    const int* rowOffset, const int* colIndices, double* matrixValues, double* rhsVector)
{
    for (int i = 0; i < 3; ++i) {
        const unsigned int vertexIndexI = *(&triangle.x + i);

        //elements of the global matrix
        const int indexOfFirstElementInRow = rowOffset[vertexIndexI];

        int numElementsInRow = rowOffset[vertexIndexI + 1] - indexOfFirstElementInRow;
        for (int j = 0; j < 3; ++j) {
            const unsigned int vertexIndexJ = *(&triangle.x + j);
            const int index = indexBinarySearch(vertexIndexJ, &colIndices[indexOfFirstElementInRow], numElementsInRow);

            if (index >= 0)
                atomicAdd(&matrixValues[index + indexOfFirstElementInRow], localMatrix(i, j) * area);
        }

        //element of the right hand side vector
        atomicAdd(&rhsVector[vertexIndexI], localRhs(i) * area);
    }
}

class NumericalIntegrator2D
{
public:
    NumericalIntegrator2D(const Mesh2D &mesh_);

    virtual ~NumericalIntegrator2D();

protected:
    const Mesh2D &mesh;

};

#endif // NUMERICAL_INTEGRATOR_3D_CUH
