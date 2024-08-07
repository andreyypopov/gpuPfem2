#include "quadrature_formula_1d.cuh"

QuadratureFormula1D::QuadratureFormula1D(int index){
    std::vector<GaussPoint1D> GaussPoints;

    std::vector<double> coordinates;
    std::vector<double> weights;

    switch (index)
    {
    case 0:
        coordinates = { 0.0 };
        weights = {	2.0 };
        break;
    case 1:
        coordinates = { -0.577350269189626, 0.577350269189626 };
        weights = {	1.0, 1.0 };
        break;
    case 2:
        coordinates = { -0.774596669241483, 0.0, 0.774596669241483 };
        weights = {	0.555555555555556, 0.888888888888889, 0.555555555555556 };
        break;
    case 3:
        coordinates = { -0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053 };
        weights = {	0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454 };
        break;
    case 4:
        coordinates = { -0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664 };
        weights = {	0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189 };
        break;
    default:
        break;
    }

    GaussPoints.reserve(coordinates.size());
    for(int i = 0; i < coordinates.size(); ++i){
        GaussPoint1D gp;
        gp.coordinate = coordinates[i];
        gp.weight = weights[i];
        GaussPoints.push_back(gp);
    }

    d_GaussPoints.allocate(GaussPoints.size());
    copy_h2d(GaussPoints.data(), d_GaussPoints.data, GaussPoints.size());
}
