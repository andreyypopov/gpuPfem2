#include "particle_2d.cuh"

Particle2D::Particle2D(const Point2 &position, const Point3 &localPosition, unsigned int ID)
    : ID(ID)
    , position(position)
    , localPosition(localPosition)
    , velocity({ 0.0, 0.0 })
{

}
