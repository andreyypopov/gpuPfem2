#ifndef MESH3D_CUH
#define MESH3D_CUH

#include "common/cuda_math.cuh"
#include "common/device_vector.cuh"

#include <string>
#include <vector>
#include <array>

class Mesh3D
{
public:
    bool loadMeshFromFile(const std::string &filename, bool fillNeighborLists = true, double scale = 1.0);

    void initMesh();

    const auto &getVertices() const {
        return vertices;
    }

    const auto &getCells() const {
        return cells;
    }

    const auto &getFaceBoundaryIDs() const {
        return faceBoundaryIDs;
    }

    const auto &getCellVolume() const {
        return cellVolume;
    }

    const auto &getInvJacobi() const {
        return invJacobi;
    }

    const auto &getCellNeighborsOffsets() const {
        return cellNeighborsOffsets;
    }

    const auto &getCellNeighborIndices() const {
        return cellNeighborIndices;
    }

    const auto &getHostVertices() const {
        return hostVertices;
    }

    const auto &getHostCells() const {
        return hostCells;
    }

private:
    void fillCellNeighborIndicesCPU(const std::vector<uint4> &hostCells);
    void fillCellNeighborIndices();

    deviceVector<Point3> vertices;               //!< Vector of vertices coordinates
    deviceVector<uint4> cells;                   //!< Vector of indices of vertices describing each tetrahedral cell
    deviceVector<int4> faceBoundaryIDs;          //!< Vector of boundary IDs for faces of each tetrahedron

    deviceVector<int> cellNeighborsOffsets;
    deviceVector<int> cellNeighborIndices;

    deviceVector<double> cellVolume;
    deviceVector<GenericMatrix3x3> invJacobi;

    std::vector<Point3> hostVertices;
    std::vector<uint4> hostCells;
};

#endif // MESH3D_CUH
