#ifndef DATA_EXPORT_H
#define DATA_EXPORT_H

#include "mesh_2d.cuh"

#include <map>
#include <vector>

class DataExport
{
public:
    explicit DataExport(const Mesh2D &mesh);

    void addScalarDataVector(const deviceVector<double> &dataVector, const std::string &fieldname);

    void exportToVTK(const std::string &filename) const;

private:
    const Mesh2D &mesh;

    std::map<std::string, double*> scalarDataVectors;
    std::map<std::string, std::vector<double>> hostScalarDataVectors;
};

#endif // DATA_EXPORT_H
