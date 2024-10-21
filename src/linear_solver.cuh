#ifndef linear_solver_cuh
#define linear_solver_cuh

#include "common/device_vector.cuh"
#include "sparse_matrix.cuh"

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse_v2.h"

#include <vector>

class LinearSolver
{
public:
    LinearSolver(double tolerance, int max_iterations);
    virtual ~LinearSolver();

    virtual void init(const SparseMatrixCSR& matrix, bool usePreconditioning = false);

    virtual bool solve(const SparseMatrixCSR& A, deviceVector<double>& x, const deviceVector<double>& b);

protected:
    //cusparse and cublas handles
    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer;
    double aSpmv, bSpmv;                //coefficients used in SPMV

    const SparseMatrixCSR *csrMatrix = nullptr;

    deviceVector<double> invDiagValues;

    double residual_norm;

    double tolerance;
    double tolerance_squared;
    int maxIterations;

    int n;
    int nnz;
    bool usePreconditioning;

    int gpuBlocks;
};

// CUDA implementation of the conventional and Chronopolous / Gear conjugate gradient method
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
class SolverCG : public LinearSolver
{
public:
    SolverCG(double tolerance, int max_iterations);
    virtual ~SolverCG();
    
    void init(const SparseMatrixCSR &matrix, bool usePreconditioning = false);

	bool solveChronopolousGear(const SparseMatrixCSR &A, deviceVector<double> &x, const deviceVector<double> &b);

    bool solve(const SparseMatrixCSR &A, deviceVector<double> &x, const deviceVector<double> &b);

private:
    //CG vectors
    deviceVector<double> rk;
    deviceVector<double> uk;
    deviceVector<double> wk;
    deviceVector<double> pk;
    deviceVector<double> sk;
    deviceVector<double> Apk;
    deviceVector<double> zk;

    //CG coefficients/scalars
    double *alpha_k;
    double *beta_k;
    double *gamma_kp, *gamma_k;
    double *delta_k;
    double *pkApk;

    const bool ChronopolousGear = false;
};

// CUDA implementation of the GMRES solver
//
// Implemented according to Saad Y. Iterative Methods for Sparse Linear Systems
//
// At this point no restart is performed
//
class SolverGMRES : public LinearSolver
{
public:
    SolverGMRES(double tolerance, int max_iterations);
    virtual ~SolverGMRES();

    void init(const SparseMatrixCSR &matrix, bool usePreconditioning = false);

    bool solve(const SparseMatrixCSR &A, deviceVector<double> &x, const deviceVector<double> &b);

private:
    /*!
     * @brief Upper-triangular part of the Hessenberg matrix H
     * 
     * Elements below diagonal are not stored. At each iteration corresponding element is temporarily stored in aux
     * and then zeroed using Givens rotation.
     * 
     * The matrix is stored in column-major format (columns are stored as rows) and additional columns are filled
     * during the progress of the algorithm
     */
    deviceVector<double> Hmatrix;

    /*!
     * @brief Matrix of Krylov subspace vectors
     * 
     * The matrix is stored in column-major format (columns are stored as rows) and additional columns are filled
     * during the progress of the algorithm
     */
    deviceVector<double> Vmatrix;
    
    deviceVector<double> cs;        //!< Vector of cosine values of the Givens rotations
    deviceVector<double> sn;        //!< Vector of sine values of the Givens rotations
    deviceVector<double> beta;      //!< Beta vector used in the linear least squares problem
    deviceVector<double> y;         //!< Solution vector of the linear least squares problem (later used in the triangular solve)

    double *v_kp, *v_k;     //!< Variables used only as pointers to different locations in the V matrix (no additional allocation)
    double *aux;            //!< Auxiliary variable for the Hessenberg matrix H
    double *d_abSpmv;       //!< Additional variable for coefficients in dense matrix-vector multiplication from Cublas
                            //!< (as the pointer mode for scalar values for Cublas is set to 'device') 
};

#endif // linear_solver_cuh
