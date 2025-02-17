#ifndef MATRIX4X4_CUH
#define MATRIX4X4_CUH

#include "cuda_math.cuh"

class Vector4
{
public:
    __host__ __device__ inline Vector4()
        : Vector4(0.0){ };

    __host__ __device__ inline Vector4(const double &value){
        for(int i = 0; i < 4; ++i)
            data[i] = value;
    };

    __host__ __device__ inline Vector4 &operator=(const double &value){
        for(int i = 0; i < 4; ++i)
            data[i] = value;

        return *this;
    }

    __host__ __device__ inline double& operator()(int i)
    {
        return data[i];
    }

    __host__ __device__ inline const double& operator()(int i) const
    {
        return data[i];
    }

private:
    double data[4];
};

class Matrix4x4
{
public:
    __host__ __device__ virtual inline double& operator()(int i, int j) = 0;
    __host__ __device__ virtual inline const double& operator()(int i, int j) const = 0;
};

class GenericMatrix4x4 : public Matrix4x4
{
public:
    __host__ __device__ inline GenericMatrix4x4()
    {
        for (int i = 0; i < 16; ++i)
            data[i] = 0.0;
    }

    __host__ __device__ inline GenericMatrix4x4(const Point4 &r1, const Point4 &r2, const Point4 &r3, const Point4 &r4)
    {
        data[0] = r1.x; data[1] = r1.y; data[2] = r1.z; data[3] = r1.w;
        data[4] = r2.x; data[5] = r2.y; data[6] = r2.z; data[7] = r2.w;
        data[8] = r3.x; data[9] = r3.y; data[10] = r3.z; data[11] = r3.w;
        data[12] = r4.x; data[13] = r4.y; data[14] = r4.z; data[15] = r4.w;
    }

    __host__ __device__ inline GenericMatrix4x4(double e11, double e12, double e13, double e14,
        double e21, double e22, double e23, double e24, double e31, double e32, double e33, double e34,
        double e41, double e42, double e43, double e44)
    {
        data[0] = e11; data[1] = e12; data[2] = e13; data[3] = e14;
        data[4] = e21; data[5] = e22; data[6] = e23; data[7] = e24;
        data[8] = e31; data[9] = e32; data[10] = e33; data[11] = e34;
        data[12] = e41; data[13] = e42; data[14] = e43; data[15] = e44;
    }

    __host__ __device__ virtual inline double& operator()(int i, int j) override
    {
        return data[4 * i + j];
    }

    __host__ __device__ virtual inline const double& operator()(int i, int j) const override
    {
        return data[4 * i + j];
    }

    __host__ __device__ inline GenericMatrix4x4 transpose() const
    {
        return GenericMatrix4x4(data[0], data[4], data[8], data[12],
                                data[1], data[5], data[9], data[13],
                                data[2], data[6], data[10], data[14],
                                data[3], data[7], data[11], data[15]);
    }

private:
    double data[16];
};

#endif // MATRIX4X4_CUH
