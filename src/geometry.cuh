#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

namespace GEOMETRY
{
    __device__ inline Point2 transformLocalToGlobal(const Point3 &Lcoordinates, const Point2 *triangleVertices){
        return Lcoordinates.x * triangleVertices[0] + Lcoordinates.y * triangleVertices[1] + Lcoordinates.z * triangleVertices[2];
    }

    __device__ inline Point3 transformGlobalToLocal(const Point2 &globalCoord, const Matrix2x2 &invJacobi, const Point2 &v3){
        Point3 res;
        //invJacobi needs to be transposed here (which means multiplication of its columns by (p - v3))
        const Point2 drv3 = globalCoord - v3;
        res.x = invJacobi(0,0) * drv3.x + invJacobi(1,0) * drv3.y;
        res.y = invJacobi(0,1) * drv3.x + invJacobi(1,1) * drv3.y;
        res.z = 1.0 - res.x - res.y;

        return res;
    }

    __device__ inline bool isPointInsideUnitTriangle(const Point3 &localCoord){
        if(localCoord.x > 1.0 + CONSTANTS::DOUBLE_MIN || localCoord.x < -CONSTANTS::DOUBLE_MIN)
            return false;
        if(localCoord.y > 1.0 + CONSTANTS::DOUBLE_MIN || localCoord.y < -CONSTANTS::DOUBLE_MIN)
            return false;
        if(localCoord.z > 1.0 + CONSTANTS::DOUBLE_MIN || localCoord.z < -CONSTANTS::DOUBLE_MIN)
            return false;
        
        return true;
    }
}

#endif // GEOMETRY_CUH
