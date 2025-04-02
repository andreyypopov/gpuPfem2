#include "particle_3d.cuh"

#include "../geometry.cuh"

Particle3D::Particle3D(const Point3 &position, const Point4 &localPosition, unsigned int ID)
    : ID(ID)
    , position(position)
    , localPosition(localPosition)
    , velocity({ 0.0, 0.0, 0.0 })
{

}

__device__ bool Particle3D::isInsideCell(const GenericMatrix3x3 &invJacobi, const Point3 &v4)
{
    const Point4 Lcoordinates = GEOMETRY::transformGlobalToLocal(getPosition(), invJacobi, v4);
    if(GEOMETRY::isPointInsideUnitTet(Lcoordinates)){
        setLocalPosition(Lcoordinates);
        return true;
    } else
        return false;
}
