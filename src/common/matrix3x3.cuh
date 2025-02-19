#ifndef MATRIX3X3_CUH
#define MATRIX3X3_CUH

#include "cuda_math.cuh"

class Vector3
{
public:
    __host__ __device__ inline Vector3()
        : Vector3(0.0){ };

    __host__ __device__ inline Vector3(const double &value){
        for(int i = 0; i < 3; ++i)
            data[i] = value;
    };

    __host__ __device__ inline Vector3 &operator=(const double &value){
        for(int i = 0; i < 3; ++i)
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
    double data[3];
};

class Matrix3x3
{
public:
    __host__ __device__ virtual inline double& operator()(int i, int j) = 0;
    __host__ __device__ virtual inline const double& operator()(int i, int j) const = 0;
};

class SymmetricMatrix3x3 : public Matrix3x3
{
public:
    __host__ __device__ inline SymmetricMatrix3x3()
    {
        for (int i = 0; i < 6; ++i)
            data[i] = 0.0;
    }

    __host__ __device__ virtual inline double& operator()(int i, int j) override
    {
        if (i == j)
            return data[i];
        else {
            return data[2 + i + j];
        }
    }

    __host__ __device__ virtual inline const double& operator()(int i, int j) const override
    {
        if (i == j)
            return data[i];
        else {
            return data[2 + i + j];
        }
    }

private:
    double data[6];
};

class GenericMatrix3x3 : public Matrix3x3
{
public:
    __host__ __device__ inline GenericMatrix3x3()
    {
        for (int i = 0; i < 9; ++i)
            data[i] = 0.0;
    }

    __host__ __device__ inline GenericMatrix3x3(const Point3 &r1, const Point3 &r2, const Point3 &r3)
    {
        data[0] = r1.x; data[1] = r1.y; data[2] = r1.z;
        data[3] = r2.x; data[4] = r2.y; data[5] = r2.z;
        data[6] = r3.x; data[7] = r3.y; data[8] = r3.z;
    }

    __host__ __device__ inline GenericMatrix3x3(double e11, double e12, double e13, double e21, double e22, double e23,
        double e31, double e32, double e33)
    {
        data[0] = e11; data[1] = e12; data[2] = e13;
        data[3] = e21; data[4] = e22; data[5] = e23;
        data[6] = e31; data[7] = e32; data[8] = e33;
    }

    __host__ __device__ virtual inline double& operator()(int i, int j) override
    {
        return data[3 * i + j];
    }

    __host__ __device__ virtual inline const double& operator()(int i, int j) const override
    {
        return data[3 * i + j];
    }

    __host__ __device__ inline Point3 operator*(const Point3 &rhs) const
    {
        Point3 res;
        res.x = data[0] * rhs.x + data[1] * rhs.y + data[2] * rhs.z;
        res.y = data[3] * rhs.x + data[4] * rhs.y + data[5] * rhs.z;
        res.z = data[6] * rhs.x + data[7] * rhs.y + data[8] * rhs.z;

        return res;
    }

    __host__ __device__ inline GenericMatrix3x3 transpose() const
    {
        return GenericMatrix3x3(data[0], data[3], data[6],
                                data[1], data[4], data[7],
                                data[2], data[5], data[8]);
    }

    __host__ __device__ inline double det() const
    {
        return (data[0] * (data[4] * data[8] - data[5] * data[7]) -
                data[1] * (data[3] * data[8] - data[5] * data[6]) +
                data[2] * (data[3] * data[7] - data[4] * data[6]));
    }

    __host__ __device__ inline GenericMatrix3x3 inverse() const
    {
        const double det = this->det();

        GenericMatrix3x3 res;
        if(fabs(det) < CONSTANTS::DOUBLE_MIN)
            return res;

        const double invDet = 1.0 / det;
        res(0,0) = invDet * (data[4] * data[8] - data[5] * data[7]);
        res(0,1) = invDet * (data[2] * data[7] - data[1] * data[8]);
        res(0,2) = invDet * (data[1] * data[5] - data[2] * data[4]);
        res(1,0) = invDet * (data[5] * data[6] - data[3] * data[8]);
        res(1,1) = invDet * (data[0] * data[8] - data[2] * data[6]);
        res(1,2) = invDet * (data[2] * data[3] - data[0] * data[5]);
        res(2,0) = invDet * (data[3] * data[7] - data[4] * data[6]);
        res(2,1) = invDet * (data[1] * data[6] - data[0] * data[7]);
        res(2,2) = invDet * (data[0] * data[4] - data[1] * data[3]);

        return res;
    }

private:
    double data[9];
};

#endif // MATRIX3X3_CUH
