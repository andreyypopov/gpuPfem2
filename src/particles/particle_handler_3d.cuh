#ifndef PARTICLE_HANDLER_3D_CUH
#define PARTICLE_HANDLER_3D_CUH

#include "particle_3d.cuh"

#include "../common/device_vector.cuh"
#include "../mesh_3d.cuh"

class ParticleHandler3D
{
public:
    ParticleHandler3D(const Mesh3D *mesh_, int cellDivisionLevel);
    ~ParticleHandler3D();

    const Particle3D *getParticles() const {
        return particles.data;
    }

    int getParticleCount() const {
        return particleCount;
    }

private:
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
};

#endif // PARTICLE_HANDLER_3D_CUH
