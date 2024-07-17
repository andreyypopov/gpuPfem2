#include "cuda_math.cuh"

#include <cstdio>

__host__ __device__ SymmetricMatrix3x3::SymmetricMatrix3x3()
{
    for(int i = 0; i < 6; ++i)
        data[i] = 0.0;
}

__host__ __device__ double &SymmetricMatrix3x3::operator()(int i, int j)
{
    if(i == j)
        return data[i];
    else {
        return data[2 + i + j];
    }
}

__host__ __device__ const double &SymmetricMatrix3x3::operator()(int i, int j) const
{
    if(i == j)
        return data[i];
    else {
        return data[2 + i + j];
    }
}

__host__ __device__ Matrix2x2::Matrix2x2()
{
    for (int i = 0; i < 4; ++i)
        data[i] = 0.0;
}

__host__ __device__ Matrix2x2 Matrix2x2::inverse() const
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

__host__ __device__ double Matrix2x2::det() const
{
    return data[0] * data[3] - data[1] * data[2];
}

__host__ __device__ Point2 Matrix2x2::operator*(const Point2 vec) const
{
    Point2 res({ 0.0, 0.0 });

    res.x = data[0] * vec.x + data[1] * vec.y;
    res.y = data[2] * vec.x + data[3] * vec.y;

    return res;
}

__host__ __device__ double& Matrix2x2::operator()(int i, int j)
{
    return data[i * 2 + j];
}

__host__ __device__ const double& Matrix2x2::operator()(int i, int j) const
{
    return data[i * 2 + j];
}
