#ifndef PARTICLE2D_CUH
#define PARTICLE2D_CUH

#include "../common/cuda_math.cuh"

class Particle2D
{
public:
    __host__ __device__ Particle2D(){ };

    __host__ __device__ Particle2D(const Point2 &position, const Point3 &localPosition, unsigned int ID = 0);

    __device__ bool isInsideCell(const Matrix2x2 &invJacobi, const Point2 &v3);

    unsigned int getID() const {
        return ID;
    }

    __host__ __device__ Point2 getPosition() const {
        return position;
    }

    __host__ __device__ void setPosition(const Point2 &newPosition){
        position = newPosition;
    }

    __host__ __device__ Point3 getLocalPosition() const {
        return localPosition;
    }

    __host__ __device__ Point2 getVelocity() const {
        return velocity;
    }

    __host__ __device__ void setVelocity(const Point2 &newVelocity){
        velocity = newVelocity;
    }

    __host__ __device__ unsigned int getCellID() const {
        return cellID;
    }

    __host__ __device__ void setCellID(unsigned int newCellID){
        cellID = newCellID;
    }

private:
    __host__ __device__ void setLocalPosition(const Point3 &newLocalPosition) {
        localPosition = newLocalPosition;
    }

    unsigned int ID;
    Point2 position;
    Point3 localPosition;
    Point2 velocity;

    unsigned int cellID;
};

#endif // PARTICLE_CUH
