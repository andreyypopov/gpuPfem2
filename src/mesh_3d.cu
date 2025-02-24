#include "mesh_3d.cuh"

#include "common/cuda_memory.cuh"

#include <cub/device/device_scan.cuh>

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

__global__ void kFindNeighbors(int n, const uint4 *cells, int *cellNeighborsOffsets, int *cellNeighborIndices = nullptr){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ uint4 sharedCells[gpuThreadsMax];

    unsigned int neighborCount = 0;
    int offset = cellNeighborsOffsets[idx];

    for(int blockStart = 0; blockStart < n; blockStart += gpuThreadsMax){
        //load batch of cell data into shared memory
        if(blockStart + threadIdx.x < n)
            sharedCells[threadIdx.x] = cells[blockStart + threadIdx.x];

        __syncthreads();

        if(idx < n){
            const uint4 tri1 = cells[idx];

            for(int cellIdx = 0; cellIdx < gpuThreadsMax; ++cellIdx)
                if(blockStart + cellIdx < n && cellIdx != idx){
                    unsigned int commonPoints = 0;

                    const uint4 tri2 = sharedCells[cellIdx];

                    for(int i = 0; i < 4; ++i)
                        for(int j = 0; j < 4; ++j)
                            if(*(&tri1.x + i) == *(&tri2.x + j))
                                ++commonPoints;

                    if(commonPoints == 3){
                        if(cellNeighborIndices)
                            cellNeighborIndices[offset + neighborCount] = blockStart + cellIdx;
                        ++neighborCount;
                    }
                }
        }

        __syncthreads();
    }

    //number of neighbors of the cell is temporarily written to the offset vector
    //(+1 position is used for the purpose of further prefix sum to convert numbers to offsets,
    //so that the last element + 1 will contain the total count of neighbors for all cells)
    if(!cellNeighborIndices && idx < n)
        cellNeighborsOffsets[idx + 1] = neighborCount;
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
            fillCellNeighborIndices();

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

void Mesh3D::fillCellNeighborIndicesCPU(const std::vector<uint4> &hostCells)
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

void Mesh3D::fillCellNeighborIndices()
{
    cellNeighborsOffsets.allocate(cells.size + 1);
    cellNeighborsOffsets.clearValues();

    //1. Get number of neighbors for each cell, without saving the neighbor indices
    unsigned int blocks = blocksForSize(cells.size, gpuThreadsMax);
    kFindNeighbors<<<blocks, gpuThreadsMax>>>(cells.size, cells.data, cellNeighborsOffsets.data);

    //2.1 Get the necessary buffer size for the prefix sum (+1 offset is used as the cell 1 contains number of neighbors for cell 0, 2 - for cell 1, etc.)
    void *tmpStorage = nullptr;
    size_t tmpStorageBytes = 0;
    cub::DeviceScan::InclusiveSum(tmpStorage, tmpStorageBytes, cellNeighborsOffsets.data + 1, cellNeighborsOffsets.data + 1, cells.size);

    //2.2 Allocate the buffer for the prefix sum
    checkCudaErrors(cudaMalloc(&tmpStorage, tmpStorageBytes));

    //2.3 Perform the prefix sum to convert numbers of neighbors to offsets (the last number will be equal to total number)
    cub::DeviceScan::InclusiveSum(tmpStorage, tmpStorageBytes, cellNeighborsOffsets.data + 1, cellNeighborsOffsets.data + 1, cells.size);

    //3. Copy the total number of neighbors for all cells and allocate the vector for neighbor indices
    int total;
    copy_d2h(cellNeighborsOffsets.data + (cellNeighborsOffsets.size - 1), &total, 1);
    cellNeighborIndices.allocate(total);

    //4. Run the kernel again, this time saving the neighbor indices and not altering the offsets
    kFindNeighbors<<<blocks, gpuThreadsMax>>>(cells.size, cells.data, cellNeighborsOffsets.data, cellNeighborIndices.data);

    checkCudaErrors(cudaFree(tmpStorage));
}
