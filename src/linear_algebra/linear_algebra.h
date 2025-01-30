#ifndef linear_algebra_cuh
#define linear_algebra_cuh

#include "cublas_v2.h"
#include "cusparse_v2.h"

#include "../common/cuda_helper.cuh"
#include "../common/device_vector.cuh"

class LinearAlgebra
{
public:
    LinearAlgebra(){
        checkCublasErrors(cublasCreate(&cublasHandle));
        checkCusparseErrors(cusparseCreate(&cusparseHandle));
        
        //needed for correct execution of functions which return scalar result (dot, nrm2)
        checkCublasErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE));        
        //checkCusparseErrors(cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE));
    };

    ~LinearAlgebra(){
        checkCublasErrors(cublasDestroy(cublasHandle));
        checkCusparseErrors(cusparseDestroy(cusparseHandle));
    };

    void dot(const deviceVector<double> &v1, const deviceVector<double> &v2, double *res) const
    {
        if(v1.size == v2.size)
            dot(v1.data, v2.data, res, v1.size);
        else
            printf("Inconsistent dimensions of vectors in dot product\n");
    }

    void dot(const double *v1, const double *v2, double *res, int n) const
    {
        checkCublasErrors(cublasDdot(cublasHandle, n, v1, 1, v2, 1, res));
    }
    
    void normSquared(const double *v, double *res, int n) const
    {
        checkCublasErrors(cublasDnrm2(cublasHandle, n, v, 1, res));
    }

    void solveGeneralTriangularSystem(const double *matrix, double *solution, int n, int lda) const
    {
        checkCublasErrors(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, matrix,
            lda, solution, 1));
    }

    void generalMV(const double *matrix, const double *vec, double *res, const double *alpha, const double *beta, int m, int n, int lda) const
    {
        checkCublasErrors(cublasDgemv(cublasHandle, CUBLAS_OP_N, m, n, alpha, matrix, lda, vec, 1, beta, res, 1));
    }

    size_t sparseMV_bufferSize(const cusparseSpMatDescr_t &matrix, const cusparseDnVecDescr_t &vec, const cusparseDnVecDescr_t &result,
        const double *alpha, const double *beta) const
    {
        size_t res;
        checkCusparseErrors(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matrix, vec,
            beta, result, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &res));

        return res;
    }

#if (CUDART_VERSION >= 12040)
    void sparseMV_preprocess(const cusparseSpMatDescr_t &matrix, const cusparseDnVecDescr_t &vec, const cusparseDnVecDescr_t &result,
        const double *alpha, const double *beta, void *buffer) const
    {
        checkCusparseErrors(cusparseSpMV_preprocess(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matrix, vec,
            beta, result, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer));
    }
#endif

    void sparseMV(const cusparseSpMatDescr_t &matrix, const cusparseDnVecDescr_t &vec, const cusparseDnVecDescr_t &result,
        const double *alpha, const double *beta, void *buffer) const
    {
        checkCusparseErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matrix, vec,
            beta, result, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer));
    }

    int incompleteLU_bufferSize(const cusparseMatDescr_t &matrix, const int *rowOffset, const int *colIndices, double *matrixValues,
        const csrilu02Info_t &iluInfo, int n, int nnz) const
    {
        int res;
        checkCusparseErrors(cusparseDcsrilu02_bufferSize(cusparseHandle, n, nnz, matrix, matrixValues,
            rowOffset, colIndices, iluInfo, &res));

        return res;
    }

    void incompleteLU(const cusparseMatDescr_t &matrix, const int *rowOffset, const int *colIndices, double *matrixValues,
        const csrilu02Info_t &iluInfo, int n, int nnz, void *buffer) const
    {
        checkCusparseErrors(cusparseDcsrilu02_analysis(cusparseHandle, n, nnz, matrix, matrixValues, rowOffset, colIndices, iluInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer));
        checkCusparseErrors(cusparseDcsrilu02(cusparseHandle, n, nnz, matrix, matrixValues, rowOffset, colIndices, iluInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer));
    }

    int incompleteCholesky_bufferSize(const cusparseMatDescr_t &matrix, const int *rowOffset, const int *colIndices, double *matrixValues,
        const csric02Info_t &icInfo, int n, int nnz) const
    {
        int res;
        checkCusparseErrors(cusparseDcsric02_bufferSize(cusparseHandle, n, nnz, matrix, matrixValues,
            rowOffset, colIndices, icInfo, &res));

        return res;
    }

    void incompleteCholesky(const cusparseMatDescr_t &matrix, const int *rowOffset, const int *colIndices, double *matrixValues,
        const csric02Info_t &icInfo, int n, int nnz, void *buffer) const
    {
        checkCusparseErrors(cusparseDcsric02_analysis(cusparseHandle, n, nnz, matrix, matrixValues, rowOffset, colIndices, icInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer));
        checkCusparseErrors(cusparseDcsric02(cusparseHandle, n, nnz, matrix, matrixValues, rowOffset, colIndices, icInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer));
    }

    size_t solveSparseTriangularSystem_bufferSize(const cusparseSpMatDescr_t &matrix, const cusparseDnVecDescr_t &vec, const cusparseDnVecDescr_t &result,
        const double *alpha, const cusparseSpSVDescr_t &spsvDescription, bool transposed = false) const
    {
        size_t res;
        const cusparseOperation_t operation = transposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
        checkCusparseErrors(cusparseSpSV_bufferSize(cusparseHandle, operation, alpha, matrix,
            vec, result, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescription, &res));

        return res;
    }

    void solveSparseTriangularSystem_analysis(const cusparseSpMatDescr_t &matrix, const cusparseDnVecDescr_t &vec, const cusparseDnVecDescr_t &result,
        const double *alpha, const cusparseSpSVDescr_t &spsvDescription, void *buffer, bool transposed = false) const
    {
        const cusparseOperation_t operation = transposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
        checkCusparseErrors(cusparseSpSV_analysis(cusparseHandle, operation, alpha,
            matrix, vec, result, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescription, buffer));
    }

    void solveSparseTriangularSystem(const cusparseSpMatDescr_t &matrix, const cusparseDnVecDescr_t &vec, const cusparseDnVecDescr_t &result,
        const double *alpha, const cusparseSpSVDescr_t &spsvDescription, bool transposed = false) const
    {
        const cusparseOperation_t operation = transposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
        checkCusparseErrors(cusparseSpSV_solve(cusparseHandle, operation, alpha,
            matrix, vec, result, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescription));
    }

private:
    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
};


#endif // linear_algebra_cuh
