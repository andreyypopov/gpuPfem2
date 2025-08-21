#include "postprocessor_3d.cuh"

#include "../integration/numerical_integrator_3d.cuh"

__global__ void kProjectVelocityGradient(int n, const uint4* cells, double* volumes, const GenericMatrix3x3* invJacobi,
    double **velocity, GenericMatrix3x3* velocityGradientSum, double* weights)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n) {
        const uint4 tet = cells[idx];
        const double volume = volumes[idx];
        const GenericMatrix3x3 cellInvJacobi = invJacobi[idx];

        //1. Calculate velocity gradient for the whole cell (as the shape function gradient is constant)
        //using nodal velocities and shape function gradient values
        GenericMatrix3x3 cellVelocityGradient;
        for (int vert = 0; vert < 4; ++vert){
            const Point3 shapeGrad = cellInvJacobi * shapeFuncGrad3D(vert);
            const unsigned int nodeIndex = *(&tet.x + vert);
            const Point3 nodeVelocity = { velocity[0][nodeIndex], velocity[1][nodeIndex], velocity[2][nodeIndex] };

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    cellVelocityGradient(i, j) += *(&nodeVelocity.x + i) * *(&shapeGrad.x + j);
        }

        //2. Project the cell velocity gradient matrix onto its nodes using volume of the cell as a weighting coefficient
        for (int vert = 0; vert < 4; ++vert) {
            const unsigned int nodeIndex = *(&tet.x + vert);

            double *velGrad = velocityGradientSum[nodeIndex].rawPointer();

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    atomicAdd(velGrad + 3 * i + j,  volume * cellVelocityGradient(i, j));
            atomicAdd(&weights[nodeIndex], volume);
        }
    }
}

__global__ void kFinalizeVelocityGradientProjection(int n, GenericMatrix3x3 *velocityGradient,
    const GenericMatrix3x3 *projectionVelocityGradient, const double *projectionWeight)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n) {
        const double invProjectionWeight = 1.0 / projectionWeight[idx];

        GenericMatrix3x3 res;
        const GenericMatrix3x3 projectionValue = projectionVelocityGradient[idx];

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                res(i, j) = projectionValue(i, j) * invProjectionWeight;

        velocityGradient[idx] = res;
    }
}

__global__ void kCalculateVorticity(int n, const GenericMatrix3x3 *velocityGradient, double** vorticity)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n) {
        const GenericMatrix3x3 nodeVelocityGradient = velocityGradient[idx];

        vorticity[0][idx] = nodeVelocityGradient(2, 1) - nodeVelocityGradient(1, 2);// dVz/dy - dVy/dz
        vorticity[1][idx] = nodeVelocityGradient(0, 2) - nodeVelocityGradient(2, 0);// dVx/dz - dVz/dx
        vorticity[2][idx] = nodeVelocityGradient(1, 0) - nodeVelocityGradient(0, 1);// dVy/dx - dVx/dy
    }
}

__global__ void kCalculateQcriterion(int n, const GenericMatrix3x3 *velocityGradient, double* qCriterion)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n) {
        const GenericMatrix3x3 nodeVelocityGradient = velocityGradient[idx];

        qCriterion[idx] = -0.5 * (nodeVelocityGradient(0, 0) * nodeVelocityGradient(0, 0) + nodeVelocityGradient(1, 1) * nodeVelocityGradient(1, 1)
            + nodeVelocityGradient(2, 2) * nodeVelocityGradient(2, 2)) - (nodeVelocityGradient(0, 1) * nodeVelocityGradient(1, 0)
            + nodeVelocityGradient(0, 2) * nodeVelocityGradient(2, 0) + nodeVelocityGradient(1, 2) * nodeVelocityGradient(2, 1));
    }
}

PostProcessor3D::PostProcessor3D(const Mesh3D &mesh_, const SimulationParameters &params, double **velocity_)
    : mesh(mesh_)
    , velocity(velocity_)
{
    if (params.calculateVorticity){
        std::vector<double*> hostVorticityPointers(3);
        
        for(int i = 0; i < 3; ++i){
            vorticity[i].allocate(mesh.getVertices().size);
            vorticity[i].clearValues();
            hostVorticityPointers[i] = vorticity[i].data;            
        }
        
        vorticityPointers.allocate(3);
        copy_h2d(hostVorticityPointers.data(), vorticityPointers.data, 3);
    }
    if (params.calculateQcriterion){
        qCriterion.allocate(mesh.getVertices().size);
        qCriterion.clearValues();
    }

    velocityGradient.allocate(mesh.getVertices().size);
    velocityGradient.clearValues();

    projectionVelocityGradient.allocate(mesh.getVertices().size);
    projectionWeight.allocate(mesh.getVertices().size);
}

void PostProcessor3D::calculate()
{
    //1. Calculate velocity gradient at cell nodes
    projectionVelocityGradient.clearValues();
    projectionWeight.clearValues();

    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kProjectVelocityGradient<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getCells().data, mesh.getCellVolume().data, mesh.getInvJacobi().data,
        velocity, projectionVelocityGradient.data, projectionWeight.data);

    blocks = blocksForSize(mesh.getVertices().size);
    kFinalizeVelocityGradientProjection<<<blocks, gpuThreads>>>(mesh.getVertices().size, velocityGradient.data,
        projectionVelocityGradient.data, projectionWeight.data);

    //2. Calculate vorticity if necessary
    if(vorticityPointers.size)
        kCalculateVorticity<<<blocks, gpuThreads>>>(mesh.getVertices().size, velocityGradient.data, vorticityPointers.data);

    //3. Calculate the Q criterion if necessary
    if(qCriterion.size)
        kCalculateQcriterion<<<blocks, gpuThreads>>>(mesh.getVertices().size, velocityGradient.data, qCriterion.data);
}
