#ifndef QUADRATURE_FORMULA_2D_CUH
#define QUADRATURE_FORMULA_2D_CUH

#include "common/cuda_helper.cuh"
#include "common/cuda_math.cuh"
#include "common/device_vector.cuh"

#include <vector>

struct GaussPoint2D {
	Point3 coordinates;
	double weight;
};

std::vector<GaussPoint2D> createFaceQuadratureFormula(int index);

#endif // QUADRATURE_FORMULA_2D_CUH
