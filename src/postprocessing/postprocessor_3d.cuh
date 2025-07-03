#ifndef POSTPROCESSOR_3D_H
#define POSTPROCESSOR_3D_H

#include "../integration/quadrature_formula_3d.cuh"
#include "../mesh_3d.cuh"
#include "../parameters.cuh"

class PostProcessor3D {
public:
    PostProcessor3D(const Mesh3D &mesh_, const SimulationParameters &params, double** velocity_);

    void calculate();

    const auto &getVorticity() const {
        return vorticityPointers;
    }

    const auto &getQcriterion() const {
        return qCriterion;
    }

private:
    const Mesh3D &mesh;

    double** velocity;

    //nodal vectors for further visualization
    std::array<deviceVector<double>, 3> vorticity;
    deviceVector<double*> vorticityPointers;
    deviceVector<double> qCriterion;

    //nodal vector of velocity gradient values
    deviceVector<GenericMatrix3x3> velocityGradient;

    //values of velocity gradient sums and weights
    deviceVector<GenericMatrix3x3> projectionVelocityGradient;
    deviceVector<double> projectionWeight;
};

#endif // POSTPROCESSOR_3D_H
