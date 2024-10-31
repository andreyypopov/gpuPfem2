#include "particle_handler_2d.cuh"

#include "../numerical_integrator_2d.cuh"

__constant__ Point3 subcellCenters[CONSTANTS::MAX_PARTICLES_PER_CELL];
__constant__ int particlesPerCell;

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
            Particle2D particle(transformLocalToGlobal(subcellCenters[i], triangleVertices), subcellCenters[i], startIndex + i);
            particle.setCellID(idx);
            particles[startIndex + i] = particle;
        }
    }
}

ParticleHandler2D::ParticleHandler2D(const Mesh2D *mesh_, int cellDivisionLevel)
    : mesh(mesh_)
{
    const int subcellsNumber = std::max(std::min(cellDivisionLevel, CONSTANTS::MAX_CELL_DIVISION_LEVEL), 1);
    const int hostParticlesPerCell = subcellsNumber * subcellsNumber;
    copy_h2const(&hostParticlesPerCell, &particlesPerCell, 1);

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
    particleCount = particleNum * mesh->getCells().size;

    allocate_device(&particleIndex, 1);
}

ParticleHandler2D::~ParticleHandler2D()
{
    free_device(particleIndex);
}

void ParticleHandler2D::seedParticles()
{
    particles.allocate(particleCount);
    unsigned int blocks = blocksForSize(mesh->getCells().size);

    zero_value_device(particleIndex, 1);
    kSeedParticlesIntoCell<<<blocks, gpuThreads>>>(mesh->getCells().size, mesh->getVertices().data, mesh->getCells().data, particles.data, particleIndex);

    cudaDeviceSynchronize();
    int particlesSeeded;
    copy_d2h(particleIndex, &particlesSeeded, 1);

    printf("Created %d particles\n", particlesSeeded);
}
