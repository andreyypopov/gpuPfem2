#ifndef BOUNDARY_LOADS_CALCULATOR_3D_H
#define BOUNDARY_LOADS_CALCULATOR_3D_H

#include "../mesh_3d.cuh"
#include "../parameters.cuh"
#include "../integration/quadrature_formula_2d.cuh"

class BoundaryLoadsCalculator3D
{
public:
    BoundaryLoadsCalculator3D(const Mesh3D& mesh_, const SimulationParameters &parameters_);
    ~BoundaryLoadsCalculator3D();

    void calculateLoads(double time, const deviceVector<double*> &velocity, const deviceVector<double> &pressure, const GaussPoint2D *faceQuadratureFormula, int faceQuadraturePointsNum);

private:
    deviceVector<int2> boundaryFaces;   //index of the tetrahedron is stored together with the index of the boundary face
    deviceVector<Point3> boundaryFaceNormals;
    deviceVector<double4> faceForces;
    double4 *totalForces;
    double4 hostTotalForces;

    int* boundaryFacesCount;
    int hostBoundaryFacesCount;
    unsigned int blocks;

    const Mesh3D& mesh;
    const SimulationParameters& parameters;

    const double coeff;

    std::ofstream forcesFile;
};

#endif // BOUNDARY_LOADS_CALCULATOR_3D_H
