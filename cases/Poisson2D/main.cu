#include "data_export.cuh"
#include "Dirichlet_bcs.cuh"
#include "linear_solver.cuh"
#include "mesh_2d.cuh"
#include "numerical_integrator_2d.cuh"
#include "sparse_matrix.cuh"

#include "common/cuda_math.cuh"
#include "common/gpu_timer.cuh"

#include <vector>

__device__ double rhsFunction(const Point2& pt) {
    return exp(-(pt.x * pt.x + 10 * pt.y * pt.y));
}

__global__ void kIntegrateOverCell(int n, const Point2 *vertices, const uint3 *cells, double *areas, Matrix2x2 *invJacobi,
    const int *rowOffset, const int *colIndices, double *matrixValues, double *rhsVector,
    const GaussPoint2D *qf_points, int qf_points_num)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        const uint3 triangle = cells[idx];

        const double lambda = 0.5;
        const double area = areas[idx];

        Point2 triangleVertices[3];
        triangleVertices[0] = vertices[triangle.x];
        triangleVertices[1] = vertices[triangle.y];
        triangleVertices[2] = vertices[triangle.z];

        Matrix2x2 cellInvJacobi = invJacobi[idx];

        SymmetricMatrix3x3 localMatrix;
        double localRhs[3] = { 0.0, 0.0, 0.0 };
        double aux;

        for(int k = 0; k < qf_points_num; ++k){
            const Point3 Lcoordinates = qf_points[k].coordinates;
            Point2 quadraturePoint = faceQuadraturePoint(Lcoordinates, triangleVertices);

            for(int i = 0; i < 3; ++i){
                for(int j = i; j < 3; ++j){
                    aux = lambda * dot(cellInvJacobi * shapeFuncGrad(i), cellInvJacobi * shapeFuncGrad(j)) * qf_points[k].weight;

                    localMatrix(i, j) += aux;
                }

                aux = rhsFunction(quadraturePoint) * *(&Lcoordinates.x + i) * qf_points[k].weight;
                localRhs[i] += aux;
            }
        }

        addLocalToGlobal(triangle, area, localMatrix, localRhs, rowOffset, colIndices, matrixValues, rhsVector);
    }
}

class PoissonIntegrator : public NumericalIntegrator2D
{
public:
    PoissonIntegrator(const Mesh2D& mesh_, const QuadratureFormula2D& qf_, const QuadratureFormula1D& edgeQf_)
        : NumericalIntegrator2D(mesh_, qf_, edgeQf_) { };

	void assembleSystem(SparseMatrixCSR &csrMatrix, deviceVector<double> &rhsVector);
};

void PoissonIntegrator::assembleSystem(SparseMatrixCSR &csrMatrix, deviceVector<double> &rhsVector)
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);

    kIntegrateOverCell<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, cellArea.data, invJacobi.data,
        csrMatrix.getRowOffset(), csrMatrix.getColIndices(), csrMatrix.getMatrixValues(), rhsVector.data,
        qf.getGaussPoints(), qf.getGaussPointsNumber());
}

int main(int argc, char *argv[]){
	GpuTimer timer;
    
    timer.start();

    Mesh2D mesh;
    if(!mesh.loadMeshFromFile("../TestProblem2.dat"))
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
            const Point2& node = vertices[i];

            if (std::fabs(node.x - (-1.0)) < CONSTANTS::DOUBLE_MIN)
                hostBcs.push_back({ i, -1.0 });
            else if (std::fabs(node.x - 1.0) < CONSTANTS::DOUBLE_MIN)
                hostBcs.push_back({ i, 1.0 });
            else if (std::fabs(node.y) < CONSTANTS::DOUBLE_MIN)
                hostBcs.push_back({ i, 0.0 });
            else if (std::fabs(node.y - 1.0) < CONSTANTS::DOUBLE_MIN)
                hostBcs.push_back({ i, 2.0 });
        }

        bcs.setupDirichletBCs(hostBcs);
    }

    timer.stop("Boundary conditions setup");

    QuadratureFormula2D qf(1);
    QuadratureFormula1D edgeQf(1);

    SparseMatrixCSR matrix(mesh);
    PoissonIntegrator integrator(mesh, qf, edgeQf);

    deviceVector<double> rhsVector;
    rhsVector.allocate(problemSize);
    zero_value_device(rhsVector.data, problemSize);

    timer.start();

    integrator.assembleSystem(matrix, rhsVector);
    bcs.applyBCs(matrix, rhsVector);

    timer.stop("Assembly of system and rhs");

    matrix.exportMatrix("matrix.dat");

    deviceVector<double> solution;
    solution.allocate(problemSize);

    timer.start();

    SolverCG cgSolver(1e-8, 1000);
    cgSolver.init(matrix, true);
    cgSolver.solve(matrix, solution, rhsVector);

    timer.stop("PCG solver");
    timer.start();

    SolverGMRES gmresSolver(1e-8, 1000);
    gmresSolver.init(matrix, true);
    gmresSolver.solve(matrix, solution, rhsVector);

    timer.stop("GMRES solver");

    solution.exportToFile("solution.dat");
    rhsVector.exportToFile("rhs.dat");

    DataExport dataExport(mesh);
    dataExport.addScalarDataVector(solution, "solution");
    dataExport.exportToVTK("solution.vtu");

    return EXIT_SUCCESS;
}
