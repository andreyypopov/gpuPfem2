#include "data_export.cuh"
#include "Dirichlet_bcs.cuh"
#include "linear_solver.cuh"
#include "mesh_2d.cuh"
#include "numerical_integrator_2d.cuh"
#include "sparse_matrix.cuh"

#include <vector>

int main(int argc, char *argv[]){
	Mesh2D mesh;
    if(!mesh.loadMeshFromFile("TestProblem2.dat"))
        return EXIT_FAILURE;

    const int problemSize = mesh.getVertices().size;

    DirichletBCs bcs(mesh);

    SparseMatrixCSR matrix(mesh);
    NumericalIntegrator2D integrator(mesh, qf2D3);

    deviceVector<double> rhsVector;
    rhsVector.allocate(problemSize);
    zero_value_device(rhsVector.data, problemSize);

    integrator.assembleSystem(matrix, rhsVector);
    bcs.applyBCs(matrix, rhsVector);

    deviceVector<double> solution;
    solution.allocate(problemSize);

    SolverCG cgSolver(1e-8, 1000);
    cgSolver.init(matrix, true);
    cgSolver.solve(matrix, solution, rhsVector);

    matrix.exportMatrix("matrix.dat");
    solution.exportToFile("solution.dat");
    rhsVector.exportToFile("rhs.dat");

    DataExport dataExport(mesh);
    dataExport.addScalarDataVector(solution, "solution");
    dataExport.exportToVTK("solution.vtu");

    return EXIT_SUCCESS;
}
