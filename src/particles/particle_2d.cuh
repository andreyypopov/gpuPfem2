#ifndef PARTICLE2D_CUH
#define PARTICLE2D_CUH

#include "../common/cuda_math.cuh"

class Particle2D
{
public:
    __host__ __device__ Particle2D(){ };

    __host__ __device__ Particle2D(const Point2 &position, const Point3 &localPosition, unsigned int ID = 0);

    unsigned int getID() const {
        return ID;
    }

    Point2 getPosition() const {
        return position;
    }

    void setPosition(const Point2 &newPosition){
        position = newPosition;
    }

    Point3 getLocalPosition() const {
        return localPosition;
    }

    void setLocalPosition(const Point3 &newLocalPosition){
        localPosition = newLocalPosition;
    }

    Point2 getVelocity() const {
        return velocity;
    }

    void setVelocity(const Point2 &newVelocity){
        velocity = newVelocity;
    }

    unsigned int getCellID() const {
        return cellID;
    }

    __host__ __device__ void setCellID(unsigned int newCellID){
        cellID = newCellID;
    }

private:
    unsigned int ID;
    Point2 position;
    Point3 localPosition;
    Point2 velocity;

    unsigned int cellID;
};

#endif // PARTICLE_CUH
