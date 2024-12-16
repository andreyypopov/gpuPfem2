#ifndef PRECONDITIONERS_CUH
#define PRECONDITIONERS_CUH

#include "sparse_matrix.cuh"

class Preconditioner
{
public:
    Preconditioner(int n_);
    virtual void initialize(const SparseMatrixCSR &csrMatrix) = 0;

    virtual void applyPreconditioner(double *dest, const double *src) = 0;

protected:
    int n;
    int gpuBlocks;

private:

};

class PreconditionerJacobi : public Preconditioner
{
public:
    PreconditionerJacobi(int n_);
    virtual void initialize(const SparseMatrixCSR &csrMatrix) override;

    virtual void applyPreconditioner(double *dest, const double *src) override;

private:
    deviceVector<double> invDiagValues;
};

#endif // PRECONDITIONERS_CUH
