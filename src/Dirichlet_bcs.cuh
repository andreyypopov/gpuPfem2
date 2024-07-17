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
    explicit DirichletBCs(const Mesh2D &mesh);

    virtual void setupDirichletBCs(const Mesh2D &mesh);

    void applyBCs(SparseMatrixCSR& matrix, deviceVector<double>& rhs);

private:
    deviceVector<DirichletNode> DirichletValues;

};

#endif // DirichletBCs_CUH
