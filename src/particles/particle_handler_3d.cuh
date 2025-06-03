#ifndef PARTICLE_HANDLER_3D_CUH
#define PARTICLE_HANDLER_3D_CUH

#include "particle_3d.cuh"

#include "../common/device_vector.cuh"
#include "../mesh_3d.cuh"
#include "../parameters.cuh"

#include <fstream>
#include <optional>

class ParticleHandler3D
{
public:
    ParticleHandler3D(const Mesh3D *mesh_, SimulationParameters &params);
    ~ParticleHandler3D();

    void seedParticles();
    void initParticleVelocity(const deviceVector<double*> &velocitySolution);

    void advectParticles(const deviceVector<double*> &velocitySolution, double timeStep, int particleSubsteps);

    void correctParticleVelocity(const deviceVector<double*> &velocitySolution, const deviceVector<double*> &velocitySolutionOld);

    void projectVelocityOntoGrid(deviceVector<double*> &velocity);

    const Particle3D *getParticles() const {
        return particles.data;
    }

    int getParticleCount() const {
        return particleCount;
    }

private:
    void sortParticlesInCells();
    void checkParticleDistribution(const deviceVector<double*> &velocitySolution);

    const Mesh3D *mesh;
        
    deviceVector<Particle3D> particles;
    int *deviceParticleCount;

    deviceVector<int> particlesForCheckInNeighborCells;
    deviceVector<int> particlesToBeDeleted;
    int *particlesForCheckInNeighborCellsCount;
    int *particlesToBeDeletedCount;
    int *particlesToBeAddedCount;

    std::array<deviceVector<double>, 3> projectionVelocity;
    deviceVector<double*> projectionVelocityPtrs;
    deviceVector<double> projectionWeights;

    deviceVector<int> particleCountInSubcells;

    int particleCount;

    std::optional<std::ofstream> particleStatisticsFile;
};

#endif // PARTICLE_HANDLER_3D_CUH
