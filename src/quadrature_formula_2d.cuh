#ifndef QUADRATURE_FORMULA_2D_CUH
#define QUADRATURE_FORMULA_2D_CUH

#include "common/cuda_helper.cuh"
#include "common/cuda_math.cuh"
#include "common/device_vector.cuh"

#include <vector>

class QuadratureFormula2D
{
public:
	explicit QuadratureFormula2D(int index);

	const Point3 *getCoordinates() const {
		return d_coordinates.data;
	}

	const double *getWeights() const {
		return d_weights.data;
	}

	int getGaussPointsNumber() const {
		return d_coordinates.size;
	}

private:
	deviceVector<Point3> d_coordinates;			//!< Coordinates of Gauss points
	deviceVector<double> d_weights;				//!< Weights of Gauss points

    int order;							//!< Order of the quadrature formula (used in Rungle rule for error check)
};

#endif // QUADRATURE_FORMULA_2D_CUH
