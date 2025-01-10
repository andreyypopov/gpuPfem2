#ifndef PRECONDITIONERS_CUH
#define PRECONDITIONERS_CUH

#include "cusparse_v2.h"

#include "linear_algebra.h"
#include "sparse_matrix.cuh"

class Preconditioner
{
public:
    Preconditioner(int n_, const LinearAlgebra *LA_);
    virtual ~Preconditioner(){};

    virtual void initialize(const SparseMatrixCSR &csrMatrix) = 0;

    virtual void applyPreconditioner(double *dest, const double *src) = 0;

protected:
    int n;
    int gpuBlocks;
    const LinearAlgebra *LA;

private:

};

class PreconditionerJacobi : public Preconditioner
{
public:
    PreconditionerJacobi(int n_, const LinearAlgebra *LA_);
    virtual ~PreconditionerJacobi(){};

    virtual void initialize(const SparseMatrixCSR &csrMatrix) override;

    virtual void applyPreconditioner(double *dest, const double *src) override;

private:
    deviceVector<double> invDiagValues;
};

class PreconditionerILU : public Preconditioner
{
public:
    PreconditionerILU(const SparseMatrixCSR &matrix, const LinearAlgebra *LA_);
    virtual ~PreconditionerILU();

    virtual void initialize(const SparseMatrixCSR &matrix) override;

    virtual void applyPreconditioner(double *dest, const double *src) override;

private:
    int nnz;
    
    void* iluBuffer;
    void* lSpsvBuffer;
    void* uSpsvBuffer;

    csrilu02Info_t iluInfo;
    cusparseSpSVDescr_t lSpsvDescription, uSpsvDescription;

    double alpha;

    cusparseMatDescr_t iluMatrix;
    deviceVector<double> matrixValues;

    deviceVector<double> auxVector;

    cusparseSpMatDescr_t lMatrix, uMatrix;
    cusparseDnVecDescr_t destVec, srcVec, auxVec;

    bool analysisRequired;
};

#endif // PRECONDITIONERS_CUH
