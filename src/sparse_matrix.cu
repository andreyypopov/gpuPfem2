#include "sparse_matrix.cuh"

#include <fstream>
#include <numeric>
#include <set>
#include <vector>

SparseMatrixCSR::SparseMatrixCSR(const Mesh2D &mesh)
{
    rows = mesh.getHostVertices().size();
    const int nCells = mesh.getHostCells().size();

    std::vector<std::set<int>> connectivity(rows);

#pragma omp parallel for
    for(int i = 0; i < rows; ++i)
        connectivity[i].insert(i);

    for(int i = 0; i < nCells; ++i){
        const uint3 tri = mesh.getHostCells()[i];

        connectivity[tri.x].insert(tri.y);
        connectivity[tri.x].insert(tri.z);
        connectivity[tri.y].insert(tri.x);
        connectivity[tri.y].insert(tri.z);
        connectivity[tri.z].insert(tri.x);
        connectivity[tri.z].insert(tri.y);
    }

    std::vector<int> elementsPerRow(rows);
#pragma omp parallel for
    for(int i = 0; i < rows; ++i)
        elementsPerRow[i] = connectivity[i].size();

    totalElements = std::accumulate(elementsPerRow.begin(), elementsPerRow.end(), 0);

    std::vector<int> hostRowOffset(rows + 1);
    std::vector<int> hostColIndices(totalElements);

    for(int i = 0; i < rows; ++i)
        hostRowOffset[i + 1] = hostRowOffset[i] + elementsPerRow[i];

#pragma omp parallel for
    for(int i = 0; i < rows; ++i)
        std::copy(connectivity[i].begin(), connectivity[i].end(), hostColIndices.begin() + hostRowOffset[i]);

    rowOffset.allocate(rows + 1);
    colIndices.allocate(totalElements);
    matrixValues.allocate(totalElements);

    copy_h2d(hostRowOffset.data(), rowOffset.data, rows + 1);
    copy_h2d(hostColIndices.data(), colIndices.data, totalElements);
    zero_value_device(matrixValues.data, totalElements);

    exportMatrixStructure("matrix.dat", hostRowOffset, hostColIndices);
}

bool SparseMatrixCSR::exportMatrix(const std::string& filename) const
{
    std::ofstream outputFile(filename.c_str());

    if (outputFile.is_open()) {
        std::vector<int> hostRowOffset(rows + 1);
        std::vector<int> hostColIndices(totalElements);
        std::vector<double> hostMatrixValues(totalElements);
        
        copy_d2h(rowOffset.data, hostRowOffset.data(), rows + 1);
        copy_d2h(colIndices.data, hostColIndices.data(), totalElements);
        copy_d2h(matrixValues.data, hostMatrixValues.data(), totalElements);

        for(int i = 0; i < rows; ++i)
            for (int j = hostRowOffset[i]; j < hostRowOffset[i + 1]; ++j)
                outputFile << i + 1 << " " << hostColIndices[j] + 1 << " " << hostMatrixValues[j] << std::endl;

        outputFile.close();
        printf("Matrix saved to %s\n", filename.c_str());

        return true;
    }
    else
        return false;
}

bool exportMatrixStructure(const std::string &filename, const std::vector<int> &rowOffset, const std::vector<int> &colIndices)
{
    std::ofstream outputFile(filename.c_str());

    if(outputFile.is_open()){
        for(int i = 0; i < rowOffset.size() - 1; ++i)
            for(int j = rowOffset[i]; j < rowOffset[i + 1]; ++j)
                outputFile << i + 1 << " " << colIndices[j] + 1 << " 1" << std::endl;

        outputFile.close();
        printf("Matrix structure saved to %s\n", filename.c_str());

        return true;
    } else
        return false;
}

SparseMatrixCOO::SparseMatrixCOO(const Mesh2D &mesh)
{
    rows = mesh.getHostVertices().size();
    const int nCells = mesh.getHostCells().size();

    //number of elements = number of diagonal elements (number of vertices = rows) + number of edges in each triangle
    //multiplied by 2 (for each edge (i,j) element (j,i) will also be present)
    totalElements = rows + 3 * nCells * 2;

    elementIndices.allocate(totalElements);
    matrixValues.allocate(totalElements);

    allocate_device(&elementCount, 1);
}

SparseMatrixCOO::~SparseMatrixCOO()
{
    free_device(elementCount);
}

void SparseMatrixCOO::resetCounter()
{
    zero_value_device(elementCount, 1);
}
