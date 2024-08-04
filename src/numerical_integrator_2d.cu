#include "numerical_integrator_2d.cuh"

#include "common/cuda_helper.cuh"
#include "common/cuda_math.cuh"
#include "common/cuda_memory.cuh"

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
