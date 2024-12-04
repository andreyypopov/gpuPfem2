#include "particle_handler_2d.cuh"

#include "../geometry.cuh"

__constant__ Point3 subcellCenters[CONSTANTS::MAX_PARTICLES_PER_CELL];
__constant__ int particlesPerCell;
__constant__ int subcellsPerDim;
__constant__ double subcellStep;

__device__ int determineSubcell(const Point3 localCoords){
    int res = 0;
    
    const int i = (1.0 - localCoords.y) * subcellsPerDim;
    for(int row = 0; row < i; ++row)
        res += 2 * row + 1;

    int j = localCoords.x * subcellsPerDim;
    res += 2 * j;

    if(j != i){
        //calculate the z coordinate of the lower right corner of the (i,j) square
        //this value is the same along the whole edge which divides the square into 2 triangles
        const double x = subcellStep * (j + 1);
        const double y = 1.0 - subcellStep * (i + 1);
        const double z = 1.0 - x - y;

        //z coordinate in the upper triangle is smaller, if the particle belongs to it, the number of cell is bigger by 1
        if(localCoords.z < z)
            ++res;
    }

    return res;
}

__global__ void kSeedParticlesIntoCell(int n, const Point2 *vertices, const uint3 *cells, Particle2D *particles, int *count){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const uint3 triangle = cells[idx];

        Point2 triangleVertices[3];
        triangleVertices[0] = vertices[triangle.x];
        triangleVertices[1] = vertices[triangle.y];
        triangleVertices[2] = vertices[triangle.z];

        int startIndex = atomicAdd(count, particlesPerCell);
        for(int i = 0; i < particlesPerCell; ++i){
            Particle2D particle(GEOMETRY::transformLocalToGlobal(subcellCenters[i], triangleVertices), subcellCenters[i], startIndex + i);
            particle.setCellID(idx);
            particles[startIndex + i] = particle;
        }
    }
}

__global__ void kAdvectParticles(int n, const uint3 *cells, Particle2D *particles, double **velocity, double timeStep){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        Particle2D &particle = particles[idx];

        const uint3 triangle = cells[particle.getCellID()];

        const Point3 localPos = particles[idx].getLocalPosition();
        Point2 advectionVelocity = { 0.0, 0.0 };
        for(int i = 0; i < 3; ++i){
            advectionVelocity.x += *(&localPos.x + i) * velocity[0][*(&triangle.x + i)];
            advectionVelocity.y += *(&localPos.x + i) * velocity[1][*(&triangle.x + i)];
        }

        particle.setPosition(particle.getPosition() + timeStep * advectionVelocity);
    }
}

__global__ void kCorrectParticleVelocity(int n, const uint3 *cells, Particle2D *particles, double **velocity, double **velocityOld = nullptr){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        Particle2D &particle = particles[idx];

        const uint3 triangle = cells[particle.getCellID()];

        const Point3 localPos = particle.getLocalPosition();
        Point2 velocityIncrement = { 0.0, 0.0 };
        for(int i = 0; i < 3; ++i){
            velocityIncrement.x += *(&localPos.x + i) * (velocity[0][*(&triangle.x + i)] - (velocityOld ? velocityOld[0][*(&triangle.x + i)] : 0.0));
            velocityIncrement.y += *(&localPos.x + i) * (velocity[1][*(&triangle.x + i)] - (velocityOld ? velocityOld[1][*(&triangle.x + i)] : 0.0));
        }

        particle.setVelocity(particle.getVelocity() + velocityIncrement);
    }
}

__global__ void kProjectParticleVelocityOntoGrid(int n, const uint3 *cells, Particle2D *particles, double **projectionVelocity, double *projectionWeights){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        Particle2D &particle = particles[idx];

        const uint3 triangle = cells[particle.getCellID()];

        const Point3 localPos = particle.getLocalPosition();
        for(int i = 0; i < 3; ++i){
            const double shapeValue = *(&localPos.x + i);
            const unsigned int index = *(&triangle.x + i);
            
            atomicAdd(&projectionVelocity[0][index], shapeValue * particle.getVelocity().x);
            atomicAdd(&projectionVelocity[1][index], shapeValue * particle.getVelocity().y);
            atomicAdd(&projectionWeights[index], shapeValue);
        }
    }
}

__global__ void kFinalizeVelocityProjection(int n, double **velocity, double **projectionVelocity, double *projectionWeights){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        for(int i = 0; i < 2; ++i)
            velocity[i][idx] = projectionVelocity[i][idx] / projectionWeights[idx];
    }
}

__global__ void kCheckParticleInCell(int n, const Point2 *vertices, const uint3 *cells, const Matrix2x2 *invJacobi, Particle2D *particles, int *particlesForCheckInNeighborsCount, int *particlesForCheckInNeighbors){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        Particle2D &particle = particles[idx];
        const uint3 triangle = cells[particle.getCellID()];

        const Point2 v3 = vertices[triangle.z];
        const Matrix2x2 cellInvJacobi = invJacobi[particle.getCellID()];

        if(!particle.isInsideCell(cellInvJacobi, v3)){
            int index = atomicAdd(particlesForCheckInNeighborsCount, 1);
            particlesForCheckInNeighbors[index] = idx;
        }
    }
}

__global__ void kCheckParticleInNeighbors(int n, const Point2 *vertices, const uint3 *cells, const Matrix2x2 *invJacobi, int *cellNeighborOffsets, int *cellNeighborIndices,
    Particle2D *particles, int *particlesForCheckInNeighbors, int *particlesToBeDeletedCount, int *particlesToBeDeleted)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        int particleIdx = particlesForCheckInNeighbors[idx];
        Particle2D &particle = particles[particleIdx];

        const unsigned int oldCellID = particle.getCellID();
        bool foundCell = false;

        for(int neighborIdx = cellNeighborOffsets[oldCellID]; neighborIdx < cellNeighborOffsets[oldCellID + 1]; ++neighborIdx){
            const int neighborID = cellNeighborIndices[neighborIdx];
            
            const Point2 v3 = vertices[cells[neighborID].z];
            const Matrix2x2 cellInvJacobi = invJacobi[neighborID];

            if(particle.isInsideCell(cellInvJacobi, v3)){
                particle.setCellID(neighborID);
                foundCell = true;
                break;
            }
        }

        if(!foundCell){
            int index = atomicAdd(particlesToBeDeletedCount, 1);
            particlesToBeDeleted[index] = particleIdx;
        }
    }
}

__global__ void kDeleteParticles(int n, Particle2D *particles, int *particleCount, int *particlesToBeDeleted)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        int particleIdx = particlesToBeDeleted[idx];
        particles[particleIdx] = particles[*particleCount - n + idx];
    }
}

__global__ void kCountParticlesInSubcells(int n, Particle2D *particles, int *particleCountInSubcells)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const Particle2D &particle = particles[idx];
        const int subcellIndex = determineSubcell(particle.getLocalPosition());
        atomicAdd(particleCountInSubcells + particle.getCellID() * particlesPerCell + subcellIndex, 1);
    }
}

__global__ void kCountParticlesToBeAdded(int n, const int *particleCountInSubcells, int *particlesToBeAddedCount)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        int particlesToBeAddedInCell = 0;
        for(int i = 0; i < particlesPerCell; ++i)
            if(particleCountInSubcells[idx * particlesPerCell + i] == 0)
                ++particlesToBeAddedInCell;

        if(particlesToBeAddedInCell)
            atomicAdd(particlesToBeAddedCount, particlesToBeAddedInCell);
    }
}

__global__ void kAddParticlesToCell(int n, const Point2 *vertices, const uint3 *cells, Particle2D *particles, int *count,
    const int *particleCountInSubcells, double **velocity)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        int particlesToBeAddedInCell = 0;
        for(int i = 0; i < particlesPerCell; ++i)
            if(particleCountInSubcells[idx * particlesPerCell + i] == 0)
                ++particlesToBeAddedInCell;

        if(particlesToBeAddedInCell){    
            const uint3 triangle = cells[idx];

            Point2 triangleVertices[3];
            triangleVertices[0] = vertices[triangle.x];
            triangleVertices[1] = vertices[triangle.y];
            triangleVertices[2] = vertices[triangle.z];

            int startIndex = atomicAdd(count, particlesToBeAddedInCell);
            int addedParticleCount = 0;
            for(int i = 0; i < particlesPerCell; ++i)
                if(particleCountInSubcells[idx * particlesPerCell + i] == 0){
                    Particle2D particle(GEOMETRY::transformLocalToGlobal(subcellCenters[i], triangleVertices), subcellCenters[i], startIndex + i);
                    particle.setCellID(idx);

                    const Point3 localPos = particle.getLocalPosition();
                    Point2 newVelocity = { 0.0, 0.0 };
                    for(int j = 0; j < 3; ++j){
                        newVelocity.x += *(&localPos.x + j) * velocity[0][*(&triangle.x + j)];
                        newVelocity.y += *(&localPos.x + j) * velocity[1][*(&triangle.x + j)];
                    }

                    particle.setVelocity(newVelocity);
                    particles[startIndex + addedParticleCount] = particle;

                    ++addedParticleCount;
                }
        }
    }
}

ParticleHandler2D::ParticleHandler2D(const Mesh2D *mesh_, int cellDivisionLevel)
    : mesh(mesh_)
{
    const int subcellsNumber = std::max(std::min(cellDivisionLevel, CONSTANTS::MAX_CELL_DIVISION_LEVEL), 1);
    const int hostParticlesPerCell = subcellsNumber * subcellsNumber;
    const double hostSubcellStep = 1.0 / subcellsNumber;
    copy_h2const(&subcellsNumber, &subcellsPerDim, 1);
    copy_h2const(&hostParticlesPerCell, &particlesPerCell, 1);
    copy_h2const(&hostSubcellStep, &subcellStep, 1);

    std::vector<Point3> hostSubcellCenters(hostParticlesPerCell);

    int particleNum = -1;
    Point3 subcellVertices[3];
    Point3 center;
    double xmin, xmax, ymin, ymax;
    const double dx = 1.0 / subcellsNumber;
    
    for(int i = 0; i < subcellsNumber; ++i)
        for(int j = 0; j < 2 * i + 1; ++j){
            xmin = (j / 2) * dx;
            xmax = xmin + dx;
            ymin = (subcellsNumber - 1 - i) * dx;
            ymax = ymin + dx;

            subcellVertices[0] = { xmin, ymax, 1.0 - xmin - ymax };
            subcellVertices[1].x = (j % 2 == 0) ? xmin : xmax;
            subcellVertices[1].y = (j % 2 == 0) ? ymin : ymax;
            subcellVertices[1].z = 1.0 - subcellVertices[1].x - subcellVertices[1].y;
            subcellVertices[2] = { xmax, ymin, 1.0 - xmax - ymin };
            
            center = CONSTANTS::ONE_THIRD * (subcellVertices[0] + subcellVertices[1] + subcellVertices[2]);

            hostSubcellCenters[++particleNum] = center;
        }

    copy_h2const(hostSubcellCenters.data(), subcellCenters, hostParticlesPerCell);
    particleCount = hostParticlesPerCell * mesh->getCells().size;

    allocate_device(&deviceParticleCount, 1);
    allocate_device(&particlesForCheckInNeighborCellsCount, 1);
    allocate_device(&particlesToBeDeletedCount, 1);
    allocate_device(&particlesToBeAddedCount, 1);

    for(int i = 0; i < 2; ++i)
        projectionVelocity[i].allocate(mesh->getVertices().size);
    projectionWeights.allocate(mesh->getVertices().size);

    projectionVelocityPtrs.allocate(2);
    double* hostProjectionVelocityPtrs[2];
    for(int i = 0; i < 2; ++i)
        hostProjectionVelocityPtrs[i] = projectionVelocity[i].data;
    
    copy_h2d(hostProjectionVelocityPtrs, projectionVelocityPtrs.data, 2);

    particleCountInSubcells.allocate(mesh->getCells().size * hostParticlesPerCell);
}

ParticleHandler2D::~ParticleHandler2D()
{
    free_device(deviceParticleCount);
    free_device(particlesForCheckInNeighborCellsCount);
    free_device(particlesToBeDeletedCount);
    free_device(particlesToBeAddedCount);
}

void ParticleHandler2D::seedParticles()
{
    particles.allocate(particleCount * CONSTANTS::MEMORY_REALLOCATION_COEFFICIENT);
    particlesForCheckInNeighborCells.allocate(particleCount);
    particlesToBeDeleted.allocate(particleCount / 10);

    unsigned int blocks = blocksForSize(mesh->getCells().size);

    zero_value_device(deviceParticleCount, 1);
    kSeedParticlesIntoCell<<<blocks, gpuThreads>>>(mesh->getCells().size, mesh->getVertices().data, mesh->getCells().data, particles.data, deviceParticleCount);

    cudaDeviceSynchronize();
    int particlesSeeded;
    copy_d2h(deviceParticleCount, &particlesSeeded, 1);

    printf("Created %d particles\n", particlesSeeded);
}

void ParticleHandler2D::initParticleVelocity(const deviceVector<double*> &velocitySolution)
{
    unsigned int blocks = blocksForSize(particleCount);
    kCorrectParticleVelocity<<<blocks, gpuThreads>>>(particleCount, mesh->getCells().data, particles.data, velocitySolution.data);
}

void ParticleHandler2D::advectParticles(const deviceVector<double *> &velocitySolution, double timeStep, int particleSubsteps)
{
    const double particleAdvectionTimeStep = timeStep / particleSubsteps;

    unsigned int blocks;
    for(int i = 0; i < particleSubsteps; ++i){
        blocks = blocksForSize(particleCount);
        kAdvectParticles<<<blocks, gpuThreads>>>(particleCount, mesh->getCells().data, particles.data, velocitySolution.data, particleAdvectionTimeStep);

        sortParticlesInCells();
    }

    checkParticleDistribution(velocitySolution);
    printf("Particle handler contains %d particles\n", particleCount);
}

void ParticleHandler2D::correctParticleVelocity(const deviceVector<double *> &velocitySolution, const deviceVector<double *> &velocitySolutionOld)
{
    unsigned int blocks = blocksForSize(particleCount);
    kCorrectParticleVelocity<<<blocks, gpuThreads>>>(particleCount, mesh->getCells().data, particles.data, velocitySolution.data, velocitySolutionOld.data);
}

void ParticleHandler2D::projectVelocityOntoGrid(deviceVector<double *> &velocity)
{
    for(int i = 0; i < 2; ++i)
        projectionVelocity[i].clearValues();
    projectionWeights.clearValues();
    
    unsigned int blocks = blocksForSize(particleCount);
    kProjectParticleVelocityOntoGrid<<<blocks, gpuThreads>>>(particleCount, mesh->getCells().data, particles.data, projectionVelocityPtrs.data, projectionWeights.data);

    blocks = blocksForSize(mesh->getVertices().size);
    kFinalizeVelocityProjection<<<blocks, gpuThreads>>>(mesh->getVertices().size, velocity.data, projectionVelocityPtrs.data, projectionWeights.data);
}

void ParticleHandler2D::sortParticlesInCells()
{
    zero_value_device(particlesForCheckInNeighborCellsCount, 1);
    zero_value_device(particlesToBeDeletedCount, 1);

    //1. Check whether each particle has stayed in the same cell as previously
    unsigned int blocks = blocksForSize(particleCount);
    kCheckParticleInCell<<<blocks, gpuThreads>>>(particleCount, mesh->getVertices().data, mesh->getCells().data, mesh->getInvJacobi().data,
        particles.data, particlesForCheckInNeighborCellsCount, particlesForCheckInNeighborCells.data);

    int hostParticlesForCheckInNeighborsCount;
    copy_d2h(particlesForCheckInNeighborCellsCount, &hostParticlesForCheckInNeighborsCount, 1);

    //2. For those particles which have left the cell check the neighboring cells
    if(hostParticlesForCheckInNeighborsCount){
        blocks = blocksForSize(hostParticlesForCheckInNeighborsCount);
        kCheckParticleInNeighbors<<<blocks, gpuThreads>>>(hostParticlesForCheckInNeighborsCount, mesh->getVertices().data, mesh->getCells().data, mesh->getInvJacobi().data,
            mesh->getCellNeighborsOffsets().data, mesh->getCellNeighborIndices().data, particles.data, particlesForCheckInNeighborCells.data,
            particlesToBeDeletedCount, particlesToBeDeleted.data);
    }
    
    int hostParticlesToBeDeletedCount;
    copy_d2h(particlesToBeDeletedCount, &hostParticlesToBeDeletedCount, 1);

    //3. Delete the particles for which a cell was not found (it is done by moving them to the end of the particle vector and reducing its length)
    if(hostParticlesToBeDeletedCount){
        blocks = blocksForSize(hostParticlesToBeDeletedCount);
        kDeleteParticles<<<blocks, gpuThreads>>>(hostParticlesToBeDeletedCount, particles.data, deviceParticleCount, particlesToBeDeleted.data);

        particleCount -= hostParticlesToBeDeletedCount;
        copy_h2d(&particleCount, deviceParticleCount, 1);
    }
}

void ParticleHandler2D::checkParticleDistribution(const deviceVector<double*>& velocitySolution)
{
    zero_value_device(particlesToBeAddedCount, 1);
    particleCountInSubcells.clearValues();

    //1. Count the number of particles in subcells of each cell
    unsigned int blocks = blocksForSize(particleCount);
    kCountParticlesInSubcells<<<blocks, gpuThreads>>>(particleCount, particles.data, particleCountInSubcells.data);

    //2. Calculate the overall number of particles to be added
    blocks = blocksForSize(mesh->getCells().size);
    kCountParticlesToBeAdded<<<blocks, gpuThreads>>>(mesh->getCells().size, particleCountInSubcells.data, particlesToBeAddedCount);

    int hostParticlesToBeAddedCount;
    copy_d2h(particlesToBeAddedCount, &hostParticlesToBeAddedCount, 1);

    //3. If there are particles to be added, resize the particle vector
    //(reallocation will be performed only if the new size exceed the previous capacity)
    //and add the particles in the centers of corresponding subcells initializing their velocity
    if(hostParticlesToBeAddedCount){
        particleCount += hostParticlesToBeAddedCount;
        particles.resize(particleCount);

        kAddParticlesToCell<<<blocks, gpuThreads>>>(mesh->getCells().size, mesh->getVertices().data, mesh->getCells().data,
            particles.data, deviceParticleCount, particleCountInSubcells.data, velocitySolution.data);
    }
}
