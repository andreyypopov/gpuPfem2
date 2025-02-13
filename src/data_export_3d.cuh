#ifndef DATA_EXPORT_3D_H
#define DATA_EXPORT_3D_H

#include "mesh_3d.cuh"

#include <map>
#include <vector>

class DataExport3D
{
public:
    DataExport3D(const Mesh3D &mesh);

    void addScalarDataVector(const deviceVector<double> &dataVector, const std::string &fieldname);

    void exportToVTK(const std::string &filename) const;

private:
    const Mesh3D &mesh;

    std::map<std::string, double*> scalarDataVectors;
    std::map<std::string, std::vector<double>> hostScalarDataVectors;
};

#endif // DATA_EXPORT_3D_H
