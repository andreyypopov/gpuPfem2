#include "data_export.cuh"
#include "Dirichlet_bcs.cuh"
#include "linear_solver.cuh"
#include "mesh_2d.cuh"
#include "numerical_integrator_2d.cuh"
#include "sparse_matrix.cuh"
#include "quadrature_formula_1d.cuh"
#include "quadrature_formula_2d.cuh"
#include "parameters.cuh"

#include "common/cuda_math.cuh"
#include "common/gpu_timer.cuh"

#include <vector>

__constant__ GaussPoint2D faceQuadratureFormula[CONSTANTS::MAX_GAUSS_POINTS];
__constant__ int faceQuadraturePointsNum;
__constant__ GaussPoint1D edgeQuadratureFormula[CONSTANTS::MAX_GAUSS_POINTS];
__constant__ int edgeQuadraturePointsNum;

__constant__ SimulationParameters simParams;

__global__ void kSetEdgeBoundaryIDs(int n, const Point2 *vertices, const uint3 *cells, int3 *edgeBoundaryIDs)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint3 triangle = cells[idx];

        Point2 triangleVertices[3];
        triangleVertices[0] = vertices[triangle.x];
        triangleVertices[1] = vertices[triangle.y];
        triangleVertices[2] = vertices[triangle.z];

        int3 res;
        for (int i = 0; i < 3; ++i) {
            Point2 start = triangleVertices[i];
            Point2 end = triangleVertices[(i + 1) % 3];

            Point2 middle = 0.5 * (start + end);

            if (std::fabs(middle.x - (-1.0)) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 0;
            else if (std::fabs(middle.x - 1.0) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 1;
            else if (std::fabs(middle.y) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 2;
            else if (std::fabs(middle.y - 1.0) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 3;

            edgeBoundaryIDs[idx] = res;
        }
    }
}

__global__ void kIntegrateVelocityPrediction(int n, const Point2 *vertices, const uint3 *cells, double *areas, Matrix2x2 *invJacobi,
    const int3 *edgeBoundaryIDs, const double **velocity, const double** velocityOld,
    const int **rowOffset, const int **colIndices, double **matrixValues, double **rhsVector)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        const uint3 triangle = cells[idx];

        const double area = areas[idx];

        Point2 triangleVertices[3];
        triangleVertices[0] = vertices[triangle.x];
        triangleVertices[1] = vertices[triangle.y];
        triangleVertices[2] = vertices[triangle.z];

        const Matrix2x2 cellInvJacobi = invJacobi[idx];

        GenericMatrix3x3 localMatrix[2];
        Vector3 localRhs[2];

        double aux;

        //integral over cell
        for (int qp = 0; qp < faceQuadraturePointsNum; ++qp) {
            const Point3 Lcoordinates = faceQuadratureFormula[qp].coordinates;

            for (int i = 0; i < 3; ++i) {
                const Point2 shapeGradI = cellInvJacobi * shapeFuncGrad(i);
                const double shapeValueI = *(&Lcoordinates.x + i);

                for (int j = 0; j < 3; ++j) {
                    const Point2 shapeGradJ = cellInvJacobi * shapeFuncGrad(j);
                    const double shapeValueJ = *(&Lcoordinates.x + j);

                    aux = simParams.rho * shapeValueI * shapeValueJ * faceQuadratureFormula[qp].weight;

                    for (int k = 0; k < 2; ++k) {
                        localMatrix[k](i, j) += aux;
                        localRhs[k](i) += aux * velocityOld[k][*(&triangle.x + j)];
                    }

                    aux = simParams.mu * simParams.dt * faceQuadratureFormula[qp].weight;
                    localMatrix[0](i, j) += aux * (shapeGradI.y * shapeGradJ.y + 4.0 / 3.0 * shapeGradI.x * shapeGradJ.x);
                    localMatrix[1](i, j) += aux * (shapeGradI.x * shapeGradJ.x + 4.0 / 3.0 * shapeGradI.y * shapeGradJ.y);

                    localRhs[0](i) -= aux * (shapeGradI.y * shapeGradJ.x - 2.0 / 3.0 * shapeGradI.x * shapeGradJ.y) * velocity[1][*(&triangle.x + j)];
                    localRhs[1](i) -= aux * (shapeGradI.x * shapeGradJ.y - 2.0 / 3.0 * shapeGradI.y * shapeGradJ.x) * velocity[0][*(&triangle.x + j)];
                }
            }
        }

        //integral over boundary faces
        const int3 boundaryIDs = edgeBoundaryIDs[idx];
        for (int edge = 0; edge < 3; ++edge) {
            const int boundaryID = *(&boundaryIDs.x + edge);
            if (boundaryID == 0 || boundaryID == 1) {
                const double normalX = (boundaryID == 0) ? -1.0 : 1.0;
                const Point2 start = triangleVertices[edge];
                const Point2 end = triangleVertices[(edge + 1) % 3];

                for (int qp = 0; qp < edgeQuadraturePointsNum; ++qp) {
                    const Point2 quadraturePoint = edgeQuadraturePoint(start, end, edgeQuadratureFormula[qp].coordinate);
                    const Point3 Lcoordinates = transformGlobalToLocal(quadraturePoint, cellInvJacobi, triangleVertices[2]);

                    aux = simParams.mu * simParams.dt * edgeQuadratureFormula[qp].weight;

                    for (int i = 0; i < 3; ++i) {
                        const double shapeValueI = *(&Lcoordinates.x + i);

                        for (int j = 0; j < 3; ++j) {
                            const Point2 shapeGradJ = cellInvJacobi * shapeFuncGrad(j);

                            localMatrix[0](i, j) -= aux * shapeValueI * (4.0 / 3.0) * shapeGradJ.x * normalX;
                            localMatrix[1](i, j) -= aux * shapeValueI * shapeGradJ.x * normalX;

                            localRhs[0](i) += aux * shapeValueI * (-2.0 / 3.0) * shapeGradJ.y * normalX * velocity[1][*(&triangle.x + j)];
                            localRhs[1](i) += aux * shapeValueI * shapeGradJ.y * normalX * velocity[0][*(&triangle.x + j)];
                        }
                    }
                }
            }
        }

        addLocalToGlobal(triangle, area, localMatrix[0], localRhs[0], rowOffset[0], colIndices[0], matrixValues[0], rhsVector[0]);
        addLocalToGlobal(triangle, area, localMatrix[1], localRhs[1], rowOffset[1], colIndices[1], matrixValues[1], rhsVector[1]);
    }
}

__global__ void kIntegratePressureEquation(int n, const Point2* vertices, const uint3* cells, double* areas, Matrix2x2* invJacobi,
    const double** velocityPrediction, const int* rowOffset, const int* colIndices, double* matrixValues, double* rhsVector)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint3 triangle = cells[idx];

        const double area = areas[idx];
        const Matrix2x2 cellInvJacobi = invJacobi[idx];

        GenericMatrix3x3 localMatrix;
        Vector3 localRhs;

        double aux;

        //integral over cell
        for (int qp = 0; qp < faceQuadraturePointsNum; ++qp) {
            const Point3 Lcoordinates = faceQuadratureFormula[qp].coordinates;

            for (int i = 0; i < 3; ++i) {
                const Point2 shapeGradI = cellInvJacobi * shapeFuncGrad(i);
                const double shapeValueI = *(&Lcoordinates.x + i);

                aux = -simParams.rho / simParams.dt * shapeValueI * faceQuadratureFormula[qp].weight;

                for (int j = 0; j < 3; ++j) {
                    const Point2 shapeGradJ = cellInvJacobi * shapeFuncGrad(j);

                    const Point2 velPredictionJ = { velocityPrediction[0][*(&triangle.x + j)], velocityPrediction[1][*(&triangle.x + j)] };

                    localMatrix(i, j) += dot(shapeGradI, shapeGradJ) * faceQuadratureFormula[qp].weight;
                    localRhs(i) += aux * dot(shapeGradJ, velPredictionJ);
                }
            }
        }

        addLocalToGlobal(triangle, area, localMatrix, localRhs, rowOffset, colIndices, matrixValues, rhsVector);
    }
}

__global__ void kIntegrateVelocityCorrection(int n, const Point2* vertices, const uint3* cells, double* areas, Matrix2x2* invJacobi,
    const double** velocityPrediction, const double* pressure, const int** rowOffset, const int** colIndices, double** matrixValues, double** rhsVector)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint3 triangle = cells[idx];

        const double area = areas[idx];
        const Matrix2x2 cellInvJacobi = invJacobi[idx];

        GenericMatrix3x3 localMatrix[2];
        Vector3 localRhs[2];

        double aux, aux2;

        //integral over cell
        for (int qp = 0; qp < faceQuadraturePointsNum; ++qp) {
            const Point3 Lcoordinates = faceQuadratureFormula[qp].coordinates;

            for (int i = 0; i < 3; ++i) {
                const double shapeValueI = *(&Lcoordinates.x + i);

                aux2 = simParams.dt * shapeValueI * faceQuadratureFormula[qp].weight;

                for (int j = 0; j < 3; ++j) {
                    const Point2 shapeGradJ = cellInvJacobi * shapeFuncGrad(j);
                    const double shapeValueJ = *(&Lcoordinates.x + j);

                    aux = simParams.rho * shapeValueI * shapeValueJ * faceQuadratureFormula[qp].weight;

                    for (int k = 0; k < 2; ++k) {
                        localMatrix[k](i, j) += aux;
                        localRhs[k](i) += aux * velocityPrediction[k][*(&triangle.x + j)] - aux2 * *(&shapeGradJ.x + k) * pressure[*(&triangle.x + j)];
                    }
                }
            }
        }

        addLocalToGlobal(triangle, area, localMatrix[0], localRhs[0], rowOffset[0], colIndices[0], matrixValues[0], rhsVector[0]);
        addLocalToGlobal(triangle, area, localMatrix[1], localRhs[1], rowOffset[1], colIndices[1], matrixValues[1], rhsVector[1]);
    }
}

class PoiseuilleFlowIntegrator : public NumericalIntegrator2D
{
public:
    PoiseuilleFlowIntegrator(const Mesh2D& mesh_)
        : NumericalIntegrator2D(mesh_) { };
    
    //setup pointers (including device ones)
    void setupVelocityPrediction(std::array<SparseMatrixCSR, 2>& csrMatrix, std::array<deviceVector<double>, 2>& rhsVector,
        const std::array<deviceVector<double>, 2>& velocity);

    void setupPressure(SparseMatrixCSR& csrMatrix, deviceVector<double>& rhsVector, deviceVector<double>& solution);

    void setupVelocityCorrection(std::array<SparseMatrixCSR, 2>& csrMatrix, std::array<deviceVector<double>, 2>& rhsVector,
        const std::array<deviceVector<double>, 2>& velocity, const std::array<deviceVector<double>, 2>& velocityOld);

    //assemble matrices and right-hand-side vectors
	void assembleVelocityPrediction();

    void assemblePressureEquation();

    void assembleVelocityCorrection();

private:
    deviceVector<const double*> velocitySolution;
    deviceVector<const double*> velocitySolutionOld;
    deviceVector<const double*> velocityPrediction;
    const double* pressure;

    deviceVector<double*> velocityPredictionRhs;
    deviceVector<double*> velocityCorrectionRhs;
    double* pressureRhs;

    deviceVector<const int*> velocityPredictionRowOffset;
    deviceVector<const int*> velocityPredictionColIndices;
    deviceVector<double*> velocityPredictionMatrixValues;

    deviceVector<const int*> velocityCorrectionRowOffset;
    deviceVector<const int*> velocityCorrectionColIndices;
    deviceVector<double*> velocityCorrectionMatrixValues;

    const int* pressureRowOffset;
    const int* pressureColIndices;
    double* pressureMatrixValues;
};

void PoiseuilleFlowIntegrator::setupVelocityPrediction(std::array<SparseMatrixCSR, 2>& csrMatrix, std::array<deviceVector<double>, 2>& rhsVector, const std::array<deviceVector<double>, 2>& velocity)
{
    const double* vel[2];
    const int* rowOffset[2];
    const int* colIndices[2];
    double* matrixValues[2];
    double* rhs[2];

    for (int i = 0; i < 2; ++i) {
        vel[i] = velocity[i].data;
        rowOffset[i] = csrMatrix[i].getRowOffset();
        colIndices[i] = csrMatrix[i].getColIndices();
        matrixValues[i] = csrMatrix[i].getMatrixValues();
        rhs[i] = rhsVector[i].data;
    }

    velocityPrediction.allocate(2);
    velocityPredictionRhs.allocate(2);
    velocityPredictionRowOffset.allocate(2);
    velocityPredictionColIndices.allocate(2);
    velocityPredictionMatrixValues.allocate(2);

    copy_h2d(vel, velocityPrediction.data, 2);
    copy_h2d(rhs, velocityPredictionRhs.data, 2);
    copy_h2d(rowOffset, velocityPredictionRowOffset.data, 2);
    copy_h2d(colIndices, velocityPredictionColIndices.data, 2);
    copy_h2d(matrixValues, velocityPredictionMatrixValues.data, 2);
}

void PoiseuilleFlowIntegrator::setupPressure(SparseMatrixCSR& csrMatrix, deviceVector<double>& rhsVector, deviceVector<double>& solution)
{
    pressure = solution.data;
    pressureRhs = rhsVector.data;
    pressureRowOffset = csrMatrix.getRowOffset();
    pressureColIndices = csrMatrix.getColIndices();
    pressureMatrixValues = csrMatrix.getMatrixValues();
}

void PoiseuilleFlowIntegrator::setupVelocityCorrection(std::array<SparseMatrixCSR, 2>& csrMatrix, std::array<deviceVector<double>, 2>& rhsVector,
    const std::array<deviceVector<double>, 2>& velocity, const std::array<deviceVector<double>, 2>& velocityOld)
{
    const double* vel[2];
    const double* velOld[2];
    const int* rowOffset[2];
    const int* colIndices[2];
    double* matrixValues[2];
    double* rhs[2];

    for (int i = 0; i < 2; ++i) {
        vel[i] = velocity[i].data;
        velOld[i] = velocityOld[i].data;
        rowOffset[i] = csrMatrix[i].getRowOffset();
        colIndices[i] = csrMatrix[i].getColIndices();
        matrixValues[i] = csrMatrix[i].getMatrixValues();
        rhs[i] = rhsVector[i].data;
    }

    velocitySolution.allocate(2);
    velocitySolutionOld.allocate(2);
    velocityCorrectionRhs.allocate(2);
    velocityCorrectionRowOffset.allocate(2);
    velocityCorrectionColIndices.allocate(2);
    velocityCorrectionMatrixValues.allocate(2);

    copy_h2d(vel, velocitySolution.data, 2);
    copy_h2d(velOld, velocitySolutionOld.data, 2);
    copy_h2d(rhs, velocityCorrectionRhs.data, 2);
    copy_h2d(rowOffset, velocityCorrectionRowOffset.data, 2);
    copy_h2d(colIndices, velocityCorrectionColIndices.data, 2);
    copy_h2d(matrixValues, velocityCorrectionMatrixValues.data, 2);
}

void PoiseuilleFlowIntegrator::assembleVelocityPrediction()
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kIntegrateVelocityPrediction<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getCellArea().data, mesh.getInvJacobi().data,
        mesh.getEdgeBoundaryIDs().data, velocitySolution.data, velocitySolutionOld.data, velocityPredictionRowOffset.data,
        velocityPredictionColIndices.data, velocityPredictionMatrixValues.data, velocityPredictionRhs.data);
}

void PoiseuilleFlowIntegrator::assemblePressureEquation()
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kIntegratePressureEquation<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getCellArea().data, mesh.getInvJacobi().data,
        velocityPrediction.data, pressureRowOffset, pressureColIndices, pressureMatrixValues, pressureRhs);
}

void PoiseuilleFlowIntegrator::assembleVelocityCorrection()
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kIntegrateVelocityCorrection<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getCellArea().data, mesh.getInvJacobi().data,
        velocityPrediction.data, pressure, velocityCorrectionRowOffset.data, velocityCorrectionColIndices.data,
        velocityCorrectionMatrixValues.data, velocityCorrectionRhs.data);
}

int main(int argc, char *argv[]){
	GpuTimer timer;
    
    timer.start();

    Mesh2D mesh;
    if(!mesh.loadMeshFromFile("../TestProblem2.dat"))
        return EXIT_FAILURE;

    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kSetEdgeBoundaryIDs<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getEdgeBoundaryIDs().data);

    timer.stop("Mesh import");

    const int problemSize = mesh.getVertices().size;

    std::array<DirichletBCs, 2> velocityBCs, velocityPredictionBCs;
    DirichletBCs pressureBCs;
    
    timer.start();

    {
        std::array<std::vector<DirichletNode>, 2> hostVelocityBCs;
        std::vector<DirichletNode> hostPressureBCs;

        const auto& vertices = mesh.getHostVertices();

        hostVelocityBCs[0].reserve(0.1 * vertices.size());
        hostVelocityBCs[1].reserve(0.1 * vertices.size());
        hostPressureBCs.reserve(0.1 * vertices.size());

        for (unsigned i = 0; i < vertices.size(); ++i) {
            const Point2& node = vertices[i];

            if (std::fabs(node.x - (-1.0)) < CONSTANTS::DOUBLE_MIN) {
                //hostVelocityBCs[0].push_back({ i, 1.0 });
                hostVelocityBCs[0].push_back({ i, 6 * node.y * (1.0 - node.y) });
                hostVelocityBCs[1].push_back({ i, 0.0 });
            } else if (std::fabs(node.x - 1.0) < CONSTANTS::DOUBLE_MIN)
                hostPressureBCs.push_back({ i, 0.0 });
            else if ((std::fabs(node.y) < CONSTANTS::DOUBLE_MIN) || (std::fabs(node.y - 1.0) < CONSTANTS::DOUBLE_MIN)) {
                hostVelocityBCs[0].push_back({i, 0.0});
                hostVelocityBCs[1].push_back({i, 0.0});
            }
        }

        for (int i = 0; i < 2; ++i) {
            velocityBCs[i].setupDirichletBCs(hostVelocityBCs[i]);
            velocityPredictionBCs[i].setupDirichletBCs(hostVelocityBCs[i]);
        }
        pressureBCs.setupDirichletBCs(hostPressureBCs);
    }

    timer.stop("Boundary conditions setup");

    const auto faceQuadratureGaussPoints = createFaceQuadratureFormula(1);
    const auto edgeQuadratureGaussPoints = createEdgeQuadratureFormula(1);
    const int faceGaussPointsNum = faceQuadratureGaussPoints.size();
    const int edgeGaussPointsNum = edgeQuadratureGaussPoints.size();
    copy_h2const(faceQuadratureGaussPoints.data(), faceQuadratureFormula, faceGaussPointsNum);
    copy_h2const(&faceGaussPointsNum, &faceQuadraturePointsNum, 1);
    copy_h2const(edgeQuadratureGaussPoints.data(), edgeQuadratureFormula, edgeGaussPointsNum);
    copy_h2const(&edgeGaussPointsNum, &edgeQuadraturePointsNum, 1);

    SimulationParameters hostParams;
    hostParams.setDefaultParameters();
    hostParams.dt = 0.01;
    hostParams.tFinal = 0.5;
    hostParams.outputFrequency = 10;
    copy_h2const(&hostParams, &simParams, 1);

    //matrices, solution and right-hand-side vectors for both component of velocity field (prediction and final ones) and pressure
    std::array<SparseMatrixCSR, 2> velocityCorrectionMatrix;
    std::array<SparseMatrixCSR, 2> velocityPredictionMatrix;
    SparseMatrixCSR pressureMatrix(mesh);

	std::array<deviceVector<double>, 2> velocitySolution;
    std::array<deviceVector<double>, 2> velocitySolutionOld;
	std::array<deviceVector<double>, 2> velocityPrediction;
	deviceVector<double> pressureSolution;

    std::array<deviceVector<double>, 2> velocityCorrectionRhs;
    std::array<deviceVector<double>, 2> velocityPredictionRhs;
    deviceVector<double> pressureRhs;
	
	for(int i = 0; i < 2; ++i){
        velocityCorrectionMatrix[i].initialize(mesh);
        velocityPredictionMatrix[i].initialize(mesh);
        velocitySolution[i].allocate(problemSize);
        velocitySolutionOld[i].allocate(problemSize);
		velocityPrediction[i].allocate(problemSize);
        velocityCorrectionRhs[i].allocate(problemSize);
        velocityPredictionRhs[i].allocate(problemSize);
	}
    pressureSolution.allocate(problemSize);
    pressureRhs.allocate(problemSize);

    //initial solution
    for (int i = 0; i < 2; ++i) {
        velocitySolution[i].clearValues();
        velocityPrediction[i].clearValues();
    }
    pressureSolution.clearValues();

    PoiseuilleFlowIntegrator integrator(mesh);
    integrator.setupVelocityPrediction(velocityPredictionMatrix, velocityPredictionRhs, velocityPrediction);
    integrator.setupPressure(pressureMatrix, pressureRhs, pressureSolution);
    integrator.setupVelocityCorrection(velocityCorrectionMatrix, velocityCorrectionRhs, velocitySolution, velocitySolutionOld);

    SolverCG cgSolver(hostParams.tolerance, hostParams.maxIterations);
    cgSolver.init(pressureMatrix, true);

    SolverGMRES gmresSolver(hostParams.tolerance, hostParams.maxIterations);
    gmresSolver.init(velocityCorrectionMatrix[0], true);

    DataExport dataExport(mesh);
    dataExport.addScalarDataVector(velocitySolution[0], "velX");
    dataExport.addScalarDataVector(velocitySolution[1], "velY");
    dataExport.addScalarDataVector(velocityPrediction[0], "velPredictionX");
    dataExport.addScalarDataVector(velocityPrediction[1], "velPredictionY");
    dataExport.addScalarDataVector(pressureSolution, "pressure");
    
    dataExport.exportToVTK("solution0.vtu");

    //time loop
    unsigned int step_number = 1;
    for (double t = 0; t < hostParams.tFinal; t += hostParams.dt, ++step_number) {
        printf("\nTime step no. %u, time = %f\n", step_number, t);

        for(int i = 0; i < 2; ++i)
            copy_d2d(velocitySolution[i].data, velocitySolutionOld[i].data, problemSize);

        for (int nOuterIter = 0; nOuterIter < 1; ++nOuterIter) {
            //assemble and solve velocity prediction equations
            timer.start();
            for (int i = 0; i < 2; ++i) {
                velocityPredictionMatrix[i].clearValues();
                velocityPredictionRhs[i].clearValues();
            }
            integrator.assembleVelocityPrediction();
            for (int i = 0; i < 2; ++i) {
                velocityBCs[i].applyBCs(velocityPredictionMatrix[i], velocityPredictionRhs[i]);
                gmresSolver.solve(velocityPredictionMatrix[i], velocityPrediction[i], velocityPredictionRhs[i]);
            }
            timer.stop("Velocity prediction");

            //assemble and solve the pressure Poisson equation
            timer.start();
            pressureMatrix.clearValues();
            pressureRhs.clearValues();
            integrator.assemblePressureEquation();
            pressureBCs.applyBCs(pressureMatrix, pressureRhs);
            cgSolver.solve(pressureMatrix, pressureSolution, pressureRhs);
            timer.stop("Pressure equation");

            //assemble and solve velocity correction equations
            timer.start();
            for (int i = 0; i < 2; ++i) {
                velocityCorrectionMatrix[i].clearValues();
                velocityCorrectionRhs[i].clearValues();
            }
            integrator.assembleVelocityCorrection();
            for (int i = 0; i < 2; ++i) {
                velocityBCs[i].applyBCs(velocityCorrectionMatrix[i], velocityCorrectionRhs[i]);
                cgSolver.solve(velocityCorrectionMatrix[i], velocitySolution[i], velocityCorrectionRhs[i]);
            }
            timer.stop("Velocity correction");
        }

        if(step_number % hostParams.outputFrequency == 0)
            dataExport.exportToVTK(std::string("solution" + std::to_string(step_number) + ".vtu"));
    }

    return EXIT_SUCCESS;
}
