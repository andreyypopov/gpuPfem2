#ifndef QUADRATURE_FORMULA_2D_CUH
#define QUADRATURE_FORMULA_2D_CUH

#include "common/cuda_helper.cuh"
#include "common/cuda_math.cuh"
#include "common/device_vector.cuh"

#include <vector>

__device__ inline Point2 faceQuadraturePoint(const Point3 &Lcoordinates, const Point2 *triangleVertices){
	return Lcoordinates.x * triangleVertices[0] + Lcoordinates.y * triangleVertices[1] + Lcoordinates.z * triangleVertices[2];
}

struct GaussPoint2D {
	Point3 coordinates;
	double weight;
};

class QuadratureFormula2D
{
public:
	explicit QuadratureFormula2D(int index);

	const GaussPoint2D *getGaussPoints() const {
		return d_GaussPoints.data;
	}

	int getGaussPointsNumber() const {
		return d_GaussPoints.size;
	}

private:
	deviceVector<GaussPoint2D> d_GaussPoints;			//!< Gauss points

    int order;							//!< Order of the quadrature formula
};

#endif // QUADRATURE_FORMULA_2D_CUH
