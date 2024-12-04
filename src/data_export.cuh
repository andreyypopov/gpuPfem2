#ifndef DATA_EXPORT_H
#define DATA_EXPORT_H

#include "mesh_2d.cuh"
#include "particles/particle_handler_2d.cuh"

#include <map>
#include <vector>

class DataExport
{
public:
    DataExport(const Mesh2D &mesh, const ParticleHandler2D *particleHandler = nullptr);

    void addScalarDataVector(const deviceVector<double> &dataVector, const std::string &fieldname);

    void exportToVTK(const std::string &filename) const;

    void exportParticlesToVTK(const std::string &filename);

private:
    const Mesh2D &mesh;
    const ParticleHandler2D *particleHandler;

    std::map<std::string, double*> scalarDataVectors;
    std::map<std::string, std::vector<double>> hostScalarDataVectors;

    std::vector<Particle2D> hostParticles;
    int particleCount;
};

#endif // DATA_EXPORT_H
