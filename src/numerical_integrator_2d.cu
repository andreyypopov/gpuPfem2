#include "numerical_integrator_2d.cuh"

#include "common/cuda_helper.cuh"
#include "common/cuda_math.cuh"
#include "common/cuda_memory.cuh"

NumericalIntegrator2D::NumericalIntegrator2D(const Mesh2D &mesh_)
    : mesh(mesh_)
{

}

NumericalIntegrator2D::~NumericalIntegrator2D()
{

}
