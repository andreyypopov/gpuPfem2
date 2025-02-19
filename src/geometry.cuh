#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

namespace GEOMETRY
{
    __device__ inline Point2 transformLocalToGlobal(const Point3 &Lcoordinates, const Point2 *triangleVertices){
        return Lcoordinates.x * triangleVertices[0] + Lcoordinates.y * triangleVertices[1] + Lcoordinates.z * triangleVertices[2];
    }

    __device__ inline Point3 transformLocalToGlobal(const Point4 &Lcoordinates, const Point3 *tetVertices){
        return Lcoordinates.x * tetVertices[0] + Lcoordinates.y * tetVertices[1] + Lcoordinates.z * tetVertices[2] + Lcoordinates.w * tetVertices[3];
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

    __device__ inline Point4 transformGlobalToLocal(const Point3 &globalCoord, const GenericMatrix3x3 &invJacobi, const Point3 &v4){
        Point4 res;
        const Point3 drv4 = globalCoord - v4;
        const Point3 aux = invJacobi * drv4;
        res.x = aux.x;
        res.y = aux.y;
        res.z = aux.z;
        res.w = 1.0 - res.x - res.y - res.z;

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

    __host__ __device__ inline double distance(const Point2 &from, const Point2 &to){
        return sqrt((from.x - to.x) * (from.x - to.x) + (from.y - to.y) * (from.y - to.y));
    }
}

#endif // GEOMETRY_CUH
