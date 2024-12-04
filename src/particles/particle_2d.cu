#include "particle_2d.cuh"

#include "../geometry.cuh"

Particle2D::Particle2D(const Point2 &position, const Point3 &localPosition, unsigned int ID)
    : ID(ID)
    , position(position)
    , localPosition(localPosition)
    , velocity({ 0.0, 0.0 })
{

}

__device__ bool Particle2D::isInsideCell(const Matrix2x2 &invJacobi, const Point2 &v3)
{
    const Point3 Lcoordinates = GEOMETRY::transformGlobalToLocal(getPosition(), invJacobi, v3);
    if(GEOMETRY::isPointInsideUnitTriangle(Lcoordinates)){
        setLocalPosition(Lcoordinates);
        return true;
    } else
        return false;
}
