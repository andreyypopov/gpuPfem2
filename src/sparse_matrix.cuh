#ifndef SPARSE_MATRIX_CUH
#define SPARSE_MATRIX_CUH

#include "common/device_vector.cuh"
#include "mesh_2d.cuh"

class SparseMatrixCSR
{
public:
    SparseMatrixCSR() = default;
    explicit SparseMatrixCSR(const Mesh2D &mesh){
        initialize(mesh);
    };

    void initialize(const Mesh2D &mesh);

    int* getRowOffset() const {
        return rowOffset.data;
    }

    int* getColIndices() const {
        return colIndices.data;
    }

    double* getMatrixValues() const {
        return matrixValues.data;
    }

    int getRows() const {
        return rows;
    }

    int getTotalElements() const {
        return totalElements;
    }

    bool exportMatrix(const std::string& filename) const;

private:
    int rows;
    int totalElements;

    //CSR data
    deviceVector<int> rowOffset;
    deviceVector<int> colIndices;
    deviceVector<double> matrixValues;
};

class SparseMatrixCOO
{
public:
    explicit SparseMatrixCOO(const Mesh2D &mesh);
    ~SparseMatrixCOO();

    unsigned int getTotalElements() const {
        return totalElements;
    }

    void resetCounter();

    uint2 *getElementIndices() {
        return elementIndices.data;
    }

    double *getMatrixValues() {
        return matrixValues.data;
    }

    unsigned int *getElementCount() {
        return elementCount;
    }

private:
    unsigned int rows;
    unsigned int totalElements;

    //COO data
    deviceVector<uint2> elementIndices;
    deviceVector<double> matrixValues;

    unsigned int *elementCount;
};

bool exportMatrixStructure(const std::string &filename, const std::vector<int> &rowOffset, const std::vector<int> &colIndices);



#endif // SPARSE_MATRIX_CUH
