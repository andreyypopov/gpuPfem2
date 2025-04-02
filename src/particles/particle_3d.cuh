#ifndef PARTICLE3D_CUH
#define PARTICLE3D_CUH

#include "../common/cuda_math.cuh"

class Particle3D
{
public:
    __host__ __device__ Particle3D(){ };

    __host__ __device__ Particle3D(const Point3 &position, const Point4 &localPosition, unsigned int ID = 0);

    __device__ bool isInsideCell(const GenericMatrix3x3 &invJacobi, const Point3 &v4);

    unsigned int getID() const {
        return ID;
    }

    __host__ __device__ Point3 getPosition() const {
        return position;
    }

    __host__ __device__ void setPosition(const Point3 &newPosition){
        position = newPosition;
    }

    __host__ __device__ Point4 getLocalPosition() const {
        return localPosition;
    }

    __host__ __device__ Point3 getVelocity() const {
        return velocity;
    }

    __host__ __device__ void setVelocity(const Point3 &newVelocity){
        velocity = newVelocity;
    }

    __host__ __device__ unsigned int getCellID() const {
        return cellID;
    }

    __host__ __device__ void setCellID(unsigned int newCellID){
        cellID = newCellID;
    }

private:
    __host__ __device__ void setLocalPosition(const Point4 &newLocalPosition) {
        localPosition = newLocalPosition;
    }

    unsigned int ID;
    Point3 position;
    Point4 localPosition;
    Point3 velocity;

    unsigned int cellID;
};

#endif // PARTICLE3D_CUH
