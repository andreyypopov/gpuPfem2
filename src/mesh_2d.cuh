#ifndef MESH2D_CUH
#define MESH2D_CUH

#include "common/cuda_math.cuh"
#include "common/device_vector.cuh"

#include <string>
#include <vector>
#include <array>

class Mesh2D
{
public:
    bool loadMeshFromFile(const std::string &filename, double scale = 1.0);

    const auto &getVertices() const {
        return vertices;
    }

    const auto &getCells() const {
        return cells;
    }

    const auto &getHostVertices() const {
        return hostVertices;
    }

    const auto &getHostCells() const {
        return hostCells;
    }

private:
    deviceVector<Point2> vertices;               //!< Vector of vertices coordinates
    deviceVector<uint3> cells;                   //!< Vector of indices of vertices describing each cell

    std::vector<Point2> hostVertices;
    std::vector<uint3> hostCells;
};

#endif // MESH3D_CUH