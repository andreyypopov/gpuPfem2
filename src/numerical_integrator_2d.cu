#include "numerical_integrator_2d.cuh"

#include "common/cuda_helper.cuh"
#include "common/cuda_math.cuh"
#include "common/cuda_memory.cuh"

__device__ Point2 shapeFuncGrad(int i){
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

__device__ double rhsFunction(const Point2& pt) {
    return exp(-(pt.x * pt.x + 10 * pt.y * pt.y));
}

__device__ void addLocalToGlobal(const uint3 &triangle, const double area, const SymmetricMatrix3x3 &localMatrix, double *localRhs,
    const int *rowOffset, const int *colIndices, double *matrixValues, double *rhsVector)
{
    for(int i = 0; i < 3; ++i){
        const unsigned int vertexIndexI = *(&triangle.x + i);

        //elements of the global matrix
        const int indexOfFirstElementInRow = rowOffset[vertexIndexI];

        int numElementsInRow = rowOffset[vertexIndexI + 1] - indexOfFirstElementInRow;
        for(int j = 0; j < 3; ++j){
            const unsigned int vertexIndexJ = *(&triangle.x + j);
            const int index = indexBinarySearch(vertexIndexJ, &colIndices[indexOfFirstElementInRow], numElementsInRow);

            if(index >= 0)
                atomicAdd(&matrixValues[index + indexOfFirstElementInRow], localMatrix(i, j) * area);
        }
        
        //element of the right hand side vector
        atomicAdd(&rhsVector[vertexIndexI], localRhs[i] * area);
    }
}

__global__ void kCalculateCellArea(int n, const Point2 *vertices, const uint3 *cells, double *areas){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const uint3 triangle = cells[idx];
        const Point2 v12 = vertices[triangle.y] - vertices[triangle.x];
        const Point2 v13 = vertices[triangle.z] - vertices[triangle.x];

        areas[idx] = fabs(cross(v12, v13)) * 0.5;
    }
}

__global__ void kCalculateInvJacobi(int n, const Point2 *vertices, const uint3 *cells, Matrix2x2 *invJacobi){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const uint3 triangle = cells[idx];
        const Point2 v13 = vertices[triangle.z] - vertices[triangle.x];
        const Point2 v23 = vertices[triangle.z] - vertices[triangle.y];

        Matrix2x2 Jacobi;
        Jacobi(0, 0) = v13.x;   Jacobi(0, 1) = v13.y;
        Jacobi(1, 0) = v23.x;   Jacobi(1, 1) = v23.y;
        
        invJacobi[idx] = Jacobi.inverse();
    }
}

__global__ void kIntegrateOverCell(int n, const Point2 *vertices, const uint3 *cells, double *areas, Matrix2x2 *invJacobi,
    const int *rowOffset, const int *colIndices, double *matrixValues, double *rhsVector,
    const Point3 *qf_coordinates, const double *qf_weights, int qf_points_num)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        const uint3 triangle = cells[idx];

        const double lambda = 0.5;
        const double area = areas[idx];

        Point2 triangleVertices[3];
        triangleVertices[0] = vertices[triangle.x];
        triangleVertices[1] = vertices[triangle.y];
        triangleVertices[2] = vertices[triangle.z];

        Matrix2x2 cellInvJacobi = invJacobi[idx];

        SymmetricMatrix3x3 localMatrix;
        double localRhs[3] = { 0.0, 0.0, 0.0 };
        double aux;

        for(int k = 0; k < qf_points_num; ++k){
            Point2 quadraturePoint = { 0.0, 0.0 };
            const Point3 Lcoordinates = qf_coordinates[k];
            for (int l = 0; l < 3; ++l)
                quadraturePoint += *(&Lcoordinates.x + l) * triangleVertices[l];

            for(int i = 0; i < 3; ++i){
                for(int j = i; j < 3; ++j){
                    aux = lambda * dot(cellInvJacobi * shapeFuncGrad(i), cellInvJacobi * shapeFuncGrad(j)) * qf_weights[k];

                    localMatrix(i, j) += aux;
                }

                aux = rhsFunction(quadraturePoint) * *(&Lcoordinates.x + i) * qf_weights[k];
                localRhs[i] += aux;
            }
        }

        addLocalToGlobal(triangle, area, localMatrix, localRhs, rowOffset, colIndices, matrixValues, rhsVector);
    }
}

NumericalIntegrator2D::NumericalIntegrator2D(const Mesh2D &mesh_, const QuadratureFormula2D &qf_)
    : mesh(mesh_)
    , qf(qf_)
{
    cellArea.allocate(mesh.getCells().size);
    invJacobi.allocate(mesh.getCells().size);
    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kCalculateCellArea<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, cellArea.data);
    kCalculateInvJacobi<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, invJacobi.data);
}

NumericalIntegrator2D::~NumericalIntegrator2D()
{

}

void NumericalIntegrator2D::assembleSystem(SparseMatrixCSR &csrMatrix, deviceVector<double> &rhsVector)
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);

    kIntegrateOverCell<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, cellArea.data, invJacobi.data,
        csrMatrix.getRowOffset(), csrMatrix.getColIndices(), csrMatrix.getMatrixValues(), rhsVector.data,
        qf.getCoordinates(), qf.getWeights(), qf.getGaussPointsNumber());
}
