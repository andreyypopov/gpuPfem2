#ifndef linear_solver_cuh
#define linear_solver_cuh

#include "common/device_vector.cuh"
#include "sparse_matrix.cuh"

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse_v2.h"

#include <vector>

// CUDA implementation of the Chronopolous / Gear conjugate gradient method
// Cublas is used for dot product, Cusparse is used for sparse matrix-vector multiplication,
// therefore the application is linked against these libraries.
//
// The solver can be used with or without Jacobi preconditioner
//
// 1. init(...) should be called only once.
// Size of the matrix and number of non-zero elements is supposed to stay constant during simulation.
// 2. solve(...) is called for CG solution, passing pointers of matrix, solution and right-hand-size vectors in the device memory.
// The initial x vector is supposed to be zero.
//
// During the CG cycle the alpha and beta coefficients (see algorithm) as initialized and updated by a kernel executed by a single CUDA core.
class SolverCG
{
public:
    SolverCG(double tolerance, int max_iterations);
    ~SolverCG();
    
    void init(const SparseMatrixCSR &matrix, bool usePreconditioning = false);

	bool solveChronopolousGear(const SparseMatrixCSR &A, deviceVector<double> &x, const deviceVector<double> &b);

    bool solve(const SparseMatrixCSR &A, deviceVector<double> &x, const deviceVector<double> &b);

private:
    //cusparse and cublas handles
    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer;
    double aSpmv, bSpmv;                //coefficients used in SPMV

    //CG vectors
    deviceVector<double> rk;
    deviceVector<double> uk;
    deviceVector<double> wk;
    deviceVector<double> pk;
    deviceVector<double> sk;
    deviceVector<double> invDiagValues;
    deviceVector<double> Apk;
    deviceVector<double> zk;

    //CG coefficients/scalars
    double *alpha_k;
    double *beta_k;
    double *gamma_kp, *gamma_k;
    double *delta_k;
    double *pkApk;
    double residual_norm;

    double tolerance;
    double tolerance_squared;
    int maxIterations;

    int n;
    int nnz;
    bool usePreconditioning;

    int gpuBlocks;

    const bool ChronopolousGear = false;
};

#endif // linear_solver_cuh
