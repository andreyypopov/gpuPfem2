#ifndef MATRIX3X3_CUH
#define MATRIX3X3_CUH

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

    __host__ __device__ virtual inline double& operator()(int i, int j) override
    {
        return data[3 * i + j];
    }

    __host__ __device__ virtual inline const double& operator()(int i, int j) const override
    {
        return data[3 * i + j];
    }

private:
    double data[9];
};

#endif // MATRIX3X3_CUH
