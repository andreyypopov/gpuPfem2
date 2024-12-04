#ifndef CUDA_MATH_CUH
#define CUDA_MATH_CUH

#include "constants.h"
#include "matrix3x3.cuh"

#include <cstdio>

typedef double2 Point2;
typedef double3 Point3;

__host__ __device__ inline double sqr(double x){
    return x * x;
}

__host__ __device__ inline double sign(double x){
    if (fabs(x) < CONSTANTS::DOUBLE_MIN)
        return 0.0;
        
    return (x > CONSTANTS::DOUBLE_MIN) ? 1.0 : -1.0;
}

__host__ __device__ inline Point2 operator+(const Point2 &v1, const Point2 &v2){
    return Point2({ v1.x + v2.x, v1.y + v2.y });
}

__host__ __device__ inline Point2 operator-(const Point2 &v1, const Point2 &v2){
    return Point2({ v1.x - v2.x, v1.y - v2.y });
}

__host__ __device__ inline Point2 operator-(const Point2 &v){
    return Point2({ -v.x, -v.y });
}

__host__ __device__ inline Point2 operator*(double a, const Point2 &v){
    return Point2({ v.x * a, v.y * a });
}

__host__ __device__ inline double dot(const Point2 &v1, const Point2 &v2){
    return v1.x * v2.x + v1.y * v2.y;
}

__host__ __device__ inline double cross(const Point2 &v1, const Point2 &v2){
    return v1.x * v2.y - v1.y * v2.x;
}

__host__ __device__ inline double vector_length(const Point2 &v){
    return sqrt(dot(v, v));
}

__host__ __device__ inline double vector_length2(const Point2 &v){
    return dot(v, v);
}

__host__ __device__ inline void operator+=(Point2 &v, const Point2 &a){
    v.x += a.x;
    v.y += a.y;
}

__host__ __device__ inline void operator*=(Point2 &v, const double &a){
    v.x *= a;
    v.y *= a;
}

__host__ __device__ inline void operator/=(Point2 &v, const double &a){
    v.x /= a;
    v.y /= a;
}

__host__ __device__ inline Point2 normalize(const Point2 &v){
    const double invOldLength = 1.0 / vector_length(v);

    Point2 res;
    res.x = v.x * invOldLength;
    res.y = v.y * invOldLength;

    return res;
}

__host__ __device__ inline double norm1(const Point2 &v){
    return fabs(v.x) + fabs(v.y);
}

__host__ __device__ inline Point3 operator+(const Point3 &v1, const Point3 &v2){
    return Point3({ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z });
}

__host__ __device__ inline Point3 operator*(double a, const Point3 &v){
    return Point3({ v.x * a, v.y * a, v.z * a });
}

__host__ __device__ inline Point2 GivensRotation(const double &v1, const double &v2){
    const double coeff = rsqrt(v1 * v1 + v2 * v2);

    Point2 res;
    res.x = v1 * coeff;
    res.y = v2 * coeff;

    return res;
}

class Matrix2x2
{
public:
    __host__ __device__ inline Matrix2x2()
    {
        for (int i = 0; i < 4; ++i)
            data[i] = 0.0;
    }

    __host__ __device__ inline Matrix2x2 inverse() const
    {
        Matrix2x2 res;

        const double detA = det();
        if (fabs(detA) < CONSTANTS::DOUBLE_MIN)
            printf("det = %f\n", detA);
        //    return res;

        const double invdet = 1.0 / detA;

        res(0, 0) = data[3] * invdet;
        res(0, 1) = -data[1] * invdet;
        res(1, 0) = -data[2] * invdet;
        res(1, 1) = data[0] * invdet;

        return res;
    }

    __host__ __device__ inline double det() const
    {
        return data[0] * data[3] - data[1] * data[2];
    }

    __host__ __device__ inline Point2 operator*(const Point2 vec) const
    {
        Point2 res({ 0.0, 0.0 });

        res.x = data[0] * vec.x + data[1] * vec.y;
        res.y = data[2] * vec.x + data[3] * vec.y;

        return res;
    }

    __host__ __device__ inline double& operator()(int i, int j)
    {
        return data[i * 2 + j];
    }

    __host__ __device__ inline const double& operator()(int i, int j) const
    {
        return data[i * 2 + j];
    }

    __host__ __device__ inline Matrix2x2 transpose() const
    {
        Matrix2x2 res;
        res(0, 0) = data[0];
        res(1, 0) = data[1];
        res(0, 1) = data[2];
        res(1, 1) = data[3];

        return res;
    }

private:
    double data[4];
};

#endif // CUDA_MATH_CUH
