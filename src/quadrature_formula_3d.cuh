#ifndef QUADRATURE_FORMULA_3D_CUH
#define QUADRATURE_FORMULA_3D_CUH

#include "common/cuda_helper.cuh"
#include "common/cuda_math.cuh"
#include "common/device_vector.cuh"

#include <vector>

struct GaussPoint3D {
	Point4 coordinates;
	double weight;
};

std::vector<GaussPoint3D> createCellQuadratureFormula(int index);

#endif // QUADRATURE_FORMULA_2D_CUH
