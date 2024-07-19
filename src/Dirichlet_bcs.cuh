#ifndef DirichletBCs_CUH
#define DirichletBCs_CUH

#include "common/device_vector.cuh"
#include "mesh_2d.cuh"
#include "sparse_matrix.cuh"

struct DirichletNode
{
    unsigned int nodeIdx;
    double bcValue;
};

class DirichletBCs
{
public:
    void setupDirichletBCs(const std::vector<DirichletNode> &hostBcs);

    void applyBCs(SparseMatrixCSR& matrix, deviceVector<double>& rhs);

private:
    deviceVector<DirichletNode> DirichletValues;

};

#endif // DirichletBCs_CUH
