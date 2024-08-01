#ifndef NUMERICAL_INTEGRATOR_3D_CUH
#define NUMERICAL_INTEGRATOR_3D_CUH

#include "common/constants.h"
#include "mesh_2d.cuh"
#include "sparse_matrix.cuh"
#include "quadrature_formula_2d.cuh"

class NumericalIntegrator2D
{
public:
    NumericalIntegrator2D(const Mesh2D &mesh_, const QuadratureFormula2D &qf_);

    virtual ~NumericalIntegrator2D();

    void assembleSystem(SparseMatrixCSR &csrMatrix, deviceVector<double> &rhsVector);

protected:
    const Mesh2D &mesh;

    deviceVector<double> cellArea;
    deviceVector<Matrix2x2> invJacobi;

    const QuadratureFormula2D &qf;
};

#endif // NUMERICAL_INTEGRATOR_3D_CUH
