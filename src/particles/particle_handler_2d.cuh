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

    const Particle2D *getParticles() const {
        return particles.data;
    }

    int getParticleCount() const {
        return particleCount;
    }

private:
    const Mesh2D *mesh;
    
    deviceVector<Particle2D> particles;
    int *particleIndex;

    int particleCount;
};

#endif // PARTICLE_HANDLER_2D_CUH
