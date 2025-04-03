#include "particle_handler_3d.cuh"

#include "../geometry.cuh"

__constant__ Point4 subtetCenters[CONSTANTS::MAX_PARTICLES_PER_TET];
__constant__ int3 subtetCubeIndices[CONSTANTS::MAX_PARTICLES_PER_TET];
__constant__ int particlesPerTet;
__constant__ int subtetsPerDim;
__constant__ double subtetStep;

__host__ __device__ Point3 baseNode(int i, int j, int k, double subcellStep){
    Point3 res = { 0, 0, 0 };
    res.x = max(min(i * subcellStep, 1.0), 0.0);
    res.y = max(min(j * subcellStep, 1.0), 0.0);
    res.z = max(min(k * subcellStep, 1.0), 0.0);

    return res;
}

__global__ void kAdvectParticles3D(int n, const uint4 *cells, Particle3D *particles, double **velocity, double timeStep){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        Particle3D &particle = particles[idx];

        const uint4 tet = cells[particle.getCellID()];

        const Point4 localPos = particles[idx].getLocalPosition();
        Point3 advectionVelocity = { 0.0, 0.0, 0.0 };
        unsigned int index;
        double shapeValue;
        for(int i = 0; i < 4; ++i){
            index = *(&tet.x + i);
            shapeValue = *(&localPos.x + i);
            advectionVelocity.x += shapeValue * velocity[0][index];
            advectionVelocity.y += shapeValue * velocity[1][index];
            advectionVelocity.z += shapeValue * velocity[2][index];
        }

        particle.setPosition(particle.getPosition() + timeStep * advectionVelocity);
    }
}

__global__ void kCorrectParticleVelocity3D(int n, const uint4 *cells, Particle3D *particles, double **velocity, double **velocityOld = nullptr){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        Particle3D &particle = particles[idx];

        const uint4 tet = cells[particle.getCellID()];

        const Point4 localPos = particle.getLocalPosition();
        Point3 velocityIncrement = { 0.0, 0.0, 0.0 };
        unsigned int index;
        double shapeValue;
        for(int i = 0; i < 4; ++i){
            index = *(&tet.x + i);
            shapeValue = *(&localPos.x + i);
            velocityIncrement.x += shapeValue * (velocity[0][index] - (velocityOld ? velocityOld[0][index] : 0.0));
            velocityIncrement.y += shapeValue * (velocity[1][index] - (velocityOld ? velocityOld[1][index] : 0.0));
            velocityIncrement.z += shapeValue * (velocity[2][index] - (velocityOld ? velocityOld[2][index] : 0.0));
        }

        particle.setVelocity(particle.getVelocity() + velocityIncrement);
    }
}

__global__ void kProjectParticleVelocityOntoGrid3D(int n, const uint4 *cells, Particle3D *particles, double **projectionVelocity, double *projectionWeights){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        Particle3D &particle = particles[idx];

        const uint4 tet = cells[particle.getCellID()];

        const Point4 localPos = particle.getLocalPosition();
        unsigned int index;
        double shapeValue;
        for(int i = 0; i < 4; ++i){
            shapeValue = *(&localPos.x + i);
            index = *(&tet.x + i);
            
            atomicAdd(&projectionVelocity[0][index], shapeValue * particle.getVelocity().x);
            atomicAdd(&projectionVelocity[1][index], shapeValue * particle.getVelocity().y);
            atomicAdd(&projectionVelocity[2][index], shapeValue * particle.getVelocity().z);
            atomicAdd(&projectionWeights[index], shapeValue);
        }
    }
}

__global__ void kFinalizeVelocityProjection3D(int n, double **velocity, double **projectionVelocity, double *projectionWeights){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        for(int i = 0; i < 3; ++i)
            velocity[i][idx] = projectionVelocity[i][idx] / projectionWeights[idx];
    }
}

ParticleHandler3D::ParticleHandler3D(const Mesh3D *mesh_, int cellDivisionLevel)
    : mesh(mesh_)
{
    const int subcellsNumber = std::max(std::min(cellDivisionLevel, CONSTANTS::MAX_CELL_DIVISION_LEVEL), 1);
    const int hostParticlesPerTet = subcellsNumber * subcellsNumber * subcellsNumber;
    const double hostSubcellStep = 1.0 / subcellsNumber;
    copy_h2const(&subcellsNumber, &subtetsPerDim, 1);
    copy_h2const(&hostParticlesPerTet, &particlesPerTet, 1);
    copy_h2const(&hostSubcellStep, &subtetStep, 1);

    std::vector<Point4> hostSubcellCenters(hostParticlesPerTet);
    std::vector<int3> hostSubcellCubeIndices(hostParticlesPerTet);

    int particleNum = -1;
    Point4 subcellVertices[4];
    Point4 center;
    Point3 lowerLeft;
    
    const Point3 unitCubeVertices[8] =
        { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } };
    const int4 unitCubeTetVertexIndices[6] =
        { { 0, 1, 2, 4 }, { 2, 4, 5, 6 }, { 1, 2, 4, 5 }, { 1, 2, 3, 5 }, {2, 3, 5, 6 }, { 3, 5, 6, 7 } };

    for(int i = 0; i < subcellsNumber; ++i)
        for(int j = 0; j < subcellsNumber; ++j)
            for(int k = 0; k < subcellsNumber; ++k){
                lowerLeft = baseNode(i, j, k, hostSubcellStep);
                
                for(int tet = 0; tet < 6; ++tet){
                    bool flag = false;

                    for(int vert = 0; vert < 4; ++vert){
                        Point3 tetVertex = lowerLeft + hostSubcellStep * unitCubeVertices[*(&unitCubeTetVertexIndices[tet].x + vert)];
                        const double w = 1.0 - tetVertex.x - tetVertex.y - tetVertex.z;
                        
                        if(w > 1.0 || w < 0.0){
                            flag = true;
                            break;
                        }

                        subcellVertices[vert] = { tetVertex.x, tetVertex.y, tetVertex.z, w };
                    }

                    if(flag)
                        continue;

                    center = 0.25 * (subcellVertices[0] + subcellVertices[1] + subcellVertices[2] + subcellVertices[3]);
                    ++particleNum;
                    hostSubcellCenters[particleNum] = center;
                    hostSubcellCubeIndices[particleNum] = { i, j, k };
                }
            }

    copy_h2const(hostSubcellCenters.data(), subtetCenters, hostParticlesPerTet);
    copy_h2const(hostSubcellCubeIndices.data(), subtetCubeIndices, hostParticlesPerTet);
    particleCount = hostParticlesPerTet * mesh->getCells().size;

    allocate_device(&deviceParticleCount, 1);
    allocate_device(&particlesForCheckInNeighborCellsCount, 1);
    allocate_device(&particlesToBeDeletedCount, 1);
    allocate_device(&particlesToBeAddedCount, 1);

    for(int i = 0; i < 3; ++i)
        projectionVelocity[i].allocate(mesh->getVertices().size);
    projectionWeights.allocate(mesh->getVertices().size);

    projectionVelocityPtrs.allocate(3);
    double* hostProjectionVelocityPtrs[3];
    for(int i = 0; i < 3; ++i)
        hostProjectionVelocityPtrs[i] = projectionVelocity[i].data;
    
    copy_h2d(hostProjectionVelocityPtrs, projectionVelocityPtrs.data, 3);

    particleCountInSubcells.allocate(mesh->getCells().size * hostParticlesPerTet);
}

ParticleHandler3D::~ParticleHandler3D()
{
    free_device(deviceParticleCount);
    free_device(particlesForCheckInNeighborCellsCount);
    free_device(particlesToBeDeletedCount);
    free_device(particlesToBeAddedCount);
}

void ParticleHandler3D::correctParticleVelocity(const deviceVector<double *> &velocitySolution, const deviceVector<double *> &velocitySolutionOld)
{
    unsigned int blocks = blocksForSize(particleCount);
    kCorrectParticleVelocity3D<<<blocks, gpuThreads>>>(particleCount, mesh->getCells().data, particles.data, velocitySolution.data, velocitySolutionOld.data);
}

void ParticleHandler3D::projectVelocityOntoGrid(deviceVector<double *> &velocity)
{
    for(int i = 0; i < 3; ++i)
        projectionVelocity[i].clearValues();
    projectionWeights.clearValues();
    
    unsigned int blocks = blocksForSize(particleCount);
    kProjectParticleVelocityOntoGrid3D<<<blocks, gpuThreads>>>(particleCount, mesh->getCells().data, particles.data, projectionVelocityPtrs.data, projectionWeights.data);

    blocks = blocksForSize(mesh->getVertices().size);
    kFinalizeVelocityProjection3D<<<blocks, gpuThreads>>>(mesh->getVertices().size, velocity.data, projectionVelocityPtrs.data, projectionWeights.data);
}
