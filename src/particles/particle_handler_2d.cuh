#ifndef PARTICLE_HANDLER_2D_CUH
#define PARTICLE_HANDLER_2D_CUH

#include "particle_2d.cuh"

#include "../common/device_vector.cuh"
#include "../mesh_2d.cuh"

class ParticleHandler2D
{
public:
    ParticleHandler2D(const Mesh2D *mesh_, int cellDivisionLevel);
    ~ParticleHandler2D();

    void seedParticles();
    void initParticleVelocity(const deviceVector<double*> &velocitySolution);

    void advectParticles(const deviceVector<double*> &velocitySolution, double timeStep, int particleSubsteps);

    void correctParticleVelocity(const deviceVector<double*> &velocitySolution, const deviceVector<double*> &velocitySolutionOld);

    void projectVelocityOntoGrid(deviceVector<double*> &velocity);

    const Particle2D *getParticles() const {
        return particles.data;
    }

    int getParticleCount() const {
        return particleCount;
    }

private:
    void sortParticlesInCells();
    void checkParticleDistribution(const deviceVector<double*> &velocitySolution);

    const Mesh2D *mesh;
    
    deviceVector<Particle2D> particles;
    int *deviceParticleCount;

    deviceVector<int> particlesForCheckInNeighborCells;
    deviceVector<int> particlesToBeDeleted;
    int *particlesForCheckInNeighborCellsCount;
    int *particlesToBeDeletedCount;
    int *particlesToBeAddedCount;

    std::array<deviceVector<double>, 2> projectionVelocity;
    deviceVector<double*> projectionVelocityPtrs;
    deviceVector<double> projectionWeights;

    deviceVector<int> particleCountInSubcells;

    int particleCount;
};

#endif // PARTICLE_HANDLER_2D_CUH
