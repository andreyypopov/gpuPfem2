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
    bool loadMeshFromFile(const std::string &filename, bool fillNeighborLists = true, double scale = 1.0);

    void initMesh();

    const auto &getVertices() const {
        return vertices;
    }

    const auto &getCells() const {
        return cells;
    }

    const auto &getEdgeBoundaryIDs() const {
        return edgeBoundaryIDs;
    }

    const auto &getCellArea() const {
        return cellArea;
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
    void fillCellNeighborIndices(const std::vector<uint3> &hostCells);

    deviceVector<Point2> vertices;               //!< Vector of vertices coordinates
    deviceVector<uint3> cells;                   //!< Vector of indices of vertices describing each cell
    deviceVector<int3> edgeBoundaryIDs;          //!< Vector of boundary IDs for edges of each triangle

    deviceVector<int> cellNeighborsOffsets;
    deviceVector<int> cellNeighborIndices;

    deviceVector<double> cellArea;
    deviceVector<Matrix2x2> invJacobi;

    std::vector<Point2> hostVertices;
    std::vector<uint3> hostCells;
};

#endif // MESH3D_CUH