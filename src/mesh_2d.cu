#include "mesh_2d.cuh"

#include "common/cuda_memory.cuh"

#include <array>
#include <fstream>
#include <set>
#include <vector>

__global__ void kCalculateCellArea(int n, const Point2 *vertices, const uint3 *cells, double *areas){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const uint3 triangle = cells[idx];
        const Point2 v12 = vertices[triangle.y] - vertices[triangle.x];
        const Point2 v13 = vertices[triangle.z] - vertices[triangle.x];

        areas[idx] = fabs(cross(v12, v13)) * 0.5;
    }
}

__global__ void kCalculateInvJacobi(int n, const Point2 *vertices, const uint3 *cells, Matrix2x2 *invJacobi){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const uint3 triangle = cells[idx];
        const Point2 v31 = vertices[triangle.x] - vertices[triangle.z];
        const Point2 v32 = vertices[triangle.y] - vertices[triangle.z];

        Matrix2x2 Jacobi;
        Jacobi(0, 0) = v31.x;   Jacobi(0, 1) = v31.y;
        Jacobi(1, 0) = v32.x;   Jacobi(1, 1) = v32.y;
        
        invJacobi[idx] = Jacobi.inverse();
    }
}

bool Mesh2D::loadMeshFromFile(const std::string &filename, bool fillNeighborLists, double scale)
{
    std::ifstream meshFile(filename);

    if(meshFile.is_open()){
        int numVertices, numCells;
        int tmp;
        float tmp2;

        meshFile >> numVertices >> numCells;

        hostVertices.reserve(numVertices);
        hostCells.reserve(numCells);

        for(int i = 0; i < numVertices; ++i){
            Point2 vertex;
            meshFile >> tmp >> vertex.x >> vertex.y >> tmp2;
            hostVertices.push_back({ scale * vertex.x, scale * vertex.y });
        }

        while(!meshFile.eof()){
            meshFile >> tmp >> tmp;
            if(tmp == 203){ //encountered a triangle
                uint3 triangle;
                meshFile >> triangle.x >> triangle.y >> triangle.z;
                
                //indices of vertices are base-1 in the imported files
                triangle.x -= 1;
                triangle.y -= 1;
                triangle.z -= 1;

                hostCells.push_back(triangle);
            } else {        //encountered an entity of another type
                numCells -= 1;
                meshFile >> tmp >> tmp;
            }
        }

        meshFile.close();

        vertices.allocate(numVertices);
        cells.allocate(numCells);
        edgeBoundaryIDs.allocate(numCells);

        copy_h2d(hostVertices.data(), vertices.data, vertices.size);
        copy_h2d(hostCells.data(), cells.data, cells.size);
        set_value_device(edgeBoundaryIDs.data, -1, cells.size);

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

void Mesh2D::initMesh()
{
    cellArea.allocate(cells.size);
    invJacobi.allocate(cells.size);
    unsigned int blocks = blocksForSize(cells.size);
    kCalculateCellArea<<<blocks, gpuThreads>>>(cells.size, vertices.data, cells.data, cellArea.data);
    kCalculateInvJacobi<<<blocks, gpuThreads>>>(cells.size, vertices.data, cells.data, invJacobi.data);
}

void Mesh2D::fillCellNeighborIndices(const std::vector<uint3> &hostCells)
{
    const int numCells = hostCells.size();
    std::vector<std::set<int>> hostCellNeighbors(numCells);

    for(int i = 0; i < numCells; ++i)
        for(int j = i + 1; j < numCells; ++j){
            const uint3 triangleI = hostCells[i];
            const uint3 triangleJ = hostCells[j];

            if(triangleI.x == triangleJ.x || triangleI.x == triangleJ.y || triangleI.x == triangleJ.z ||
                triangleI.y == triangleJ.x || triangleI.y == triangleJ.y || triangleI.y == triangleJ.z ||
                triangleI.z == triangleJ.x || triangleI.z == triangleJ.y || triangleI.z == triangleJ.z){
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
