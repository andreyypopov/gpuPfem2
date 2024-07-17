#include "mesh_2d.cuh"

#include "common/cuda_memory.cuh"

#include <array>
#include <fstream>
#include <vector>

bool Mesh2D::loadMeshFromFile(const std::string &filename, double scale)
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

        copy_h2d(hostVertices.data(), vertices.data, vertices.size);
        copy_h2d(hostCells.data(), cells.data, cells.size);

        printf("Loaded mesh with %d vertices and %d cells\n", numVertices, numCells);

        return true;
    } else {
        printf("Error while opening the file\n");
        return false;
    }
}
