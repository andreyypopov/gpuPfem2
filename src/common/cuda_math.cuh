#ifndef CUDA_MATH_CUH
#define CUDA_MATH_CUH

#include "constants.h"

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

class SymmetricMatrix3x3
{
public:
    __host__ __device__ SymmetricMatrix3x3();

    __host__ __device__ double &operator()(int i, int j);
    __host__ __device__ const double &operator()(int i, int j) const;

private:
    double data[6];
};

class Matrix2x2
{
public:
    __host__ __device__ Matrix2x2();

    __host__ __device__ Matrix2x2 inverse() const;

    __host__ __device__ double det() const;

    __host__ __device__ Point2 operator*(const Point2 vec) const;

    __host__ __device__ double& operator()(int i, int j);
    __host__ __device__ const double& operator()(int i, int j) const;

private:
    double data[4];
};

#endif // CUDA_MATH_CUH
