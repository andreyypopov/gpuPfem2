#include "mesh_3d.cuh"

#include "common/cuda_memory.cuh"

#include <array>
#include <fstream>
#include <set>
#include <vector>

__global__ void kCalculateCellVolume(int n, const Point3 *vertices, const uint4 *cells, double *volumes){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const uint4 tetrahedron = cells[idx];
        const Point3 v41 = vertices[tetrahedron.x] - vertices[tetrahedron.w];
        const Point3 v42 = vertices[tetrahedron.y] - vertices[tetrahedron.w];
        const Point3 v43 = vertices[tetrahedron.z] - vertices[tetrahedron.w];

        volumes[idx] = fabs(dot(v41, cross(v42, v43))) * CONSTANTS::ONE_SIXTH;
    }
}

__global__ void kCalculateInvJacobi(int n, const Point3 *vertices, const uint4 *cells, GenericMatrix3x3 *invJacobi){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const uint4 tetrahedron = cells[idx];
        const Point3 v41 = vertices[tetrahedron.x] - vertices[tetrahedron.w];
        const Point3 v42 = vertices[tetrahedron.y] - vertices[tetrahedron.w];
        const Point3 v43 = vertices[tetrahedron.z] - vertices[tetrahedron.w];

        GenericMatrix3x3 Jacobi(v41, v42, v43);
        invJacobi[idx] = Jacobi.inverse();
    }
}

bool Mesh3D::loadMeshFromFile(const std::string &filename, bool fillNeighborLists, double scale)
{
    std::ifstream meshFile(filename);

    if(meshFile.is_open()){
        int numVertices, numCells;
        int tmp;

        meshFile >> numVertices >> numCells;

        hostVertices.reserve(numVertices);
        hostCells.reserve(numCells);

        for(int i = 0; i < numVertices; ++i){
            Point3 vertex;
            meshFile >> tmp >> vertex.x >> vertex.y >> vertex.z;
            hostVertices.push_back({ scale * vertex.x, scale * vertex.y, scale * vertex.z });
        }

        while(!meshFile.eof()){
            meshFile >> tmp >> tmp;
            if(tmp == 304){ //encountered a tetrahedron
                uint4 tetrahedron;
                meshFile >> tetrahedron.x >> tetrahedron.y >> tetrahedron.z >> tetrahedron.w;
                
                //indices of vertices are base-1 in the imported files
                tetrahedron.x -= 1;
                tetrahedron.y -= 1;
                tetrahedron.z -= 1;
                tetrahedron.w -= 1;

                hostCells.push_back(tetrahedron);
            } else {        //encountered an entity of another type
                numCells -= 1;
                if(tmp == 102)
                    meshFile >> tmp >> tmp;
                else if(tmp == 203)
                    meshFile >> tmp >> tmp >> tmp;
            }
        }

        meshFile.close();

        vertices.allocate(numVertices);
        cells.allocate(numCells);
        faceBoundaryIDs.allocate(numCells);

        copy_h2d(hostVertices.data(), vertices.data, vertices.size);
        copy_h2d(hostCells.data(), cells.data, cells.size);
        set_value_device(faceBoundaryIDs.data, -1, cells.size);

        if(fillNeighborLists)
            fillCellNeighborIndices(hostCells);

        initMesh();

        printf("Loaded mesh with %d vertices and %d cells\n", numVertices, numCells);

        return true;
    } else {
        printf("Error while opening the file\n");
        return false;
    }
}

void Mesh3D::initMesh()
{
    cellVolume.allocate(cells.size);
    invJacobi.allocate(cells.size);
    unsigned int blocks = blocksForSize(cells.size);
    kCalculateCellVolume<<<blocks, gpuThreads>>>(cells.size, vertices.data, cells.data, cellVolume.data);
    kCalculateInvJacobi<<<blocks, gpuThreads>>>(cells.size, vertices.data, cells.data, invJacobi.data);
}

void Mesh3D::fillCellNeighborIndices(const std::vector<uint4> &hostCells)
{
    const int numCells = hostCells.size();
    std::vector<std::set<int>> hostCellNeighbors(numCells);

    for(int i = 0; i < numCells; ++i)
        for(int j = i + 1; j < numCells; ++j){
            const uint4 tetI = hostCells[i];
            const uint4 tetJ = hostCells[j];

            unsigned int commonVerticesCount = 0;
            for(int vi = 0; vi < 4; ++vi)
                for(int vj = 0; vj < 4; ++vj)
                    if(*(&tetI.x + vi) == *(&tetJ.x + vj)){
                        ++commonVerticesCount;
                        break;
                    }

            if(commonVerticesCount == 3){
                hostCellNeighbors[i].insert(j);
                hostCellNeighbors[j].insert(i);
            }
        }

    std::vector<int> hostCellNeighborOffsets(numCells + 1);
    hostCellNeighborOffsets[0] = 0;
    for(int i = 0; i < numCells; ++i)
        hostCellNeighborOffsets[i + 1] = hostCellNeighborOffsets[i] + hostCellNeighbors[i].size();

    std::vector<int> hostCellNeighborIndices(hostCellNeighborOffsets.back());
    for(int i = 0; i < numCells; ++i)
        std::copy(hostCellNeighbors[i].begin(), hostCellNeighbors[i].end(), hostCellNeighborIndices.begin() + hostCellNeighborOffsets[i]);

    cellNeighborsOffsets.allocate(numCells + 1);
    cellNeighborIndices.allocate(hostCellNeighborOffsets.back());

    copy_h2d(hostCellNeighborOffsets.data(), cellNeighborsOffsets.data, numCells + 1);
    copy_h2d(hostCellNeighborIndices.data(), cellNeighborIndices.data, hostCellNeighborOffsets.back());
}
