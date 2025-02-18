#ifndef QUADRATURE_FORMULA_1D_CUH
#define QUADRATURE_FORMULA_1D_CUH

#include "../common/cuda_helper.cuh"
#include "../common/cuda_math.cuh"
#include "../common/device_vector.cuh"

#include <vector>

//qp = (b-a)/2 * x_i + (a+b)/2
__device__ inline Point2 edgeQuadraturePoint(const Point2 &start, const Point2& end, double coord){
    return 0.5 * (coord * (end - start) + (start + end));
}

struct GaussPoint1D {
	double coordinate;
	double weight;
};

std::vector<GaussPoint1D> createEdgeQuadratureFormula(int index);

#endif // QUADRATURE_FORMULA_1D_CUH
