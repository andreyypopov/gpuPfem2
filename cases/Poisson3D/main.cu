#include "data_export_3d.cuh"
#include "Dirichlet_bcs.cuh"
#include "geometry.cuh"
#include "mesh_3d.cuh"
#include "quadrature_formula_3d.cuh"
#include "numerical_integrator_3d.cuh"

#include "common/cuda_math.cuh"
#include "common/gpu_timer.cuh"

#include "linear_algebra/linear_algebra.h"
#include "linear_algebra/linear_solver.cuh"
#include "linear_algebra/preconditioners.cuh"
#include "linear_algebra/sparse_matrix.cuh"

#include <vector>

__constant__ GaussPoint3D cellQuadratureFormula[CONSTANTS::MAX_GAUSS_POINTS_3D];
__constant__ int cellQuadraturePointsNum;

__device__ double rhsFunction(const Point3& pt) {
    return exp(-(pt.x * pt.x + 10 * pt.y * pt.y + pt.z * pt.z));
}

__global__ void kIntegrateOverCell(int n, const Point3* vertices, const uint4* cells, double* volumes, GenericMatrix3x3* invJacobi,
    const int* rowOffset, const int* colIndices, double* matrixValues, double* rhsVector)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint4 tet = cells[idx];

        const double lambda = 0.5;
        const double volume = volumes[idx];

        Point3 tetVertices[4];
        tetVertices[0] = vertices[tet.x];
        tetVertices[1] = vertices[tet.y];
        tetVertices[2] = vertices[tet.z];
        tetVertices[3] = vertices[tet.w];

        const GenericMatrix3x3 tetInvJacobi = invJacobi[idx];

        GenericMatrix4x4 localMatrix;
        Vector4 localRhs;
        double aux;

        for (int k = 0; k < cellQuadraturePointsNum; ++k) {
            const Point4 Lcoordinates = cellQuadratureFormula[k].coordinates;
            const Point3 quadraturePoint = GEOMETRY::transformLocalToGlobal(Lcoordinates, tetVertices);

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    aux = lambda * dot(tetInvJacobi * shapeFuncGrad3D(i), tetInvJacobi * shapeFuncGrad3D(j)) * cellQuadratureFormula[k].weight;

                    localMatrix(i, j) += aux;
                }

                aux = rhsFunction(quadraturePoint) * *(&Lcoordinates.x + i) * cellQuadratureFormula[k].weight;
                localRhs(i) += aux;
            }
        }

        addLocalToGlobal3D(tet, volume, localMatrix, localRhs, rowOffset, colIndices, matrixValues, rhsVector);
    }
}

class PoissonIntegrator : public NumericalIntegrator3D
{
public:
    PoissonIntegrator(const Mesh3D& mesh_)
        : NumericalIntegrator3D(mesh_) {
    };

    void assembleSystem(SparseMatrixCSR& csrMatrix, deviceVector<double>& rhsVector);
};

void PoissonIntegrator::assembleSystem(SparseMatrixCSR& csrMatrix, deviceVector<double>& rhsVector)
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);

    kIntegrateOverCell<<<blocks, gpuThreads>>> (mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getCellVolume().data, mesh.getInvJacobi().data,
        csrMatrix.getRowOffset(), csrMatrix.getColIndices(), csrMatrix.getMatrixValues(), rhsVector.data);
}

int main(int argc, char *argv[]){
    GpuTimer timer;

    timer.start();

    Mesh3D mesh;
    if(!mesh.loadMeshFromFile("../BoxMesh.dat"))
        return EXIT_FAILURE;

    timer.stop("Mesh import");

    const int problemSize = mesh.getVertices().size;

    DirichletBCs bcs;

    timer.start();

    {
        std::vector<DirichletNode> hostBcs;

        const auto& vertices = mesh.getHostVertices();

        hostBcs.reserve(0.1 * vertices.size());

        for (unsigned i = 0; i < vertices.size(); ++i) {
            const Point3& node = vertices[i];

            if (std::fabs(node.x - (-1.0)) < CONSTANTS::DOUBLE_MIN)
                hostBcs.push_back({ i, -1.0 });
            else if (std::fabs(node.x - 1.0) < CONSTANTS::DOUBLE_MIN)
                hostBcs.push_back({ i, 1.0 });
            else if (std::fabs(node.y) < CONSTANTS::DOUBLE_MIN)
                hostBcs.push_back({ i, 0.0 });
            else if (std::fabs(node.y - 1.0) < CONSTANTS::DOUBLE_MIN)
                hostBcs.push_back({ i, 2.0 });
            else if (std::fabs(node.z - (-0.25)) < CONSTANTS::DOUBLE_MIN)
                hostBcs.push_back({ i, 3.0 });
            else if (std::fabs(node.z - 0.25) < CONSTANTS::DOUBLE_MIN)
                hostBcs.push_back({ i, -2.0 });
        }

        bcs.setupDirichletBCs(hostBcs);
    }

    timer.stop("Boundary conditions setup");

    const auto cellQuadratureGaussPoints = createCellQuadratureFormula(1);
    const int cellGaussPointsNum = cellQuadratureGaussPoints.size();
    copy_h2const(cellQuadratureGaussPoints.data(), cellQuadratureFormula, cellGaussPointsNum);
    copy_h2const(&cellGaussPointsNum, &cellQuadraturePointsNum, 1);

    SparseMatrixCSR matrix(mesh);
    PoissonIntegrator integrator(mesh);

    deviceVector<double> rhsVector;
    rhsVector.allocate(problemSize);
    rhsVector.clearValues();

    timer.start();

    integrator.assembleSystem(matrix, rhsVector);
    bcs.applyBCs(matrix, rhsVector);

    timer.stop("Assembly of system and rhs");

    matrix.exportMatrix("matrix.dat");

    deviceVector<double> solution;
    solution.allocate(problemSize);

    timer.start();

    LinearAlgebra LA;

    PreconditionerJacobi precond(problemSize, &LA);
    SolverCG cgSolver(1e-8, 1000, &LA, &precond);
    cgSolver.init(matrix);
    cgSolver.solve(matrix, solution, rhsVector);

    timer.stop("PCG solver");

    solution.exportToFile("solution.dat");
    rhsVector.exportToFile("rhs.dat");

    DataExport3D dataExport(mesh);
    dataExport.addScalarDataVector(solution, "solution");
    dataExport.exportToVTK("solution.vtu");

    return EXIT_SUCCESS;
}
