#include "data_export.cuh"
#include "Dirichlet_bcs.cuh"
#include "geometry.cuh"
#include "mesh_2d.cuh"
#include "numerical_integrator_2d.cuh"
#include "quadrature_formula_1d.cuh"
#include "quadrature_formula_2d.cuh"
#include "parameters.cuh"

#include "common/cuda_math.cuh"
#include "common/gpu_timer.cuh"
#include "common/profiling.h"
#include "common/utilities.h"

#include "linear_algebra/linear_algebra.h"
#include "linear_algebra/linear_solver.cuh"
#include "linear_algebra/sparse_matrix.cuh"

#include "particles/particle_handler_2d.cuh"

#include <vector>

__constant__ GaussPoint2D faceQuadratureFormula[CONSTANTS::MAX_GAUSS_POINTS];
__constant__ int faceQuadraturePointsNum;
__constant__ GaussPoint1D edgeQuadratureFormula[CONSTANTS::MAX_GAUSS_POINTS];
__constant__ int edgeQuadraturePointsNum;

__constant__ SimulationParameters simParams;

const Point2 cylinderCenter = { 0.2, 0.2 };
const double h = 0.41;
const double l = 2.2;

__global__ void kSetEdgeBoundaryIDs(int n, const Point2 *vertices, const uint3 *cells, int3 *edgeBoundaryIDs)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint3 triangle = cells[idx];

        Point2 triangleVertices[3];
        triangleVertices[0] = vertices[triangle.x];
        triangleVertices[1] = vertices[triangle.y];
        triangleVertices[2] = vertices[triangle.z];

        const Point2 cylinderCenter = { 0.2, 0.2 };
        const double h = 0.41;
        const double l = 2.2;

        int3 res = { -1, -1, -1 };
        for (int i = 0; i < 3; ++i) {
            Point2 start = triangleVertices[i];
            Point2 end = triangleVertices[(i + 1) % 3];

            Point2 middle = 0.5 * (start + end);

            if (std::fabs(middle.x) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 0;
            else if (std::fabs(middle.x - l) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 1;
            else if (std::fabs(middle.y) < CONSTANTS::DOUBLE_MIN || std::fabs(middle.y - h) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 2;
            else if (std::fabs(GEOMETRY::distance(middle, cylinderCenter) - 0.05) < 0.001)
                *(&res.x + i) = 3;

            edgeBoundaryIDs[idx] = res;
        }
    }
}

__global__ void kIntegrateVelocityPrediction(int n, const Point2 *vertices, const uint3 *cells, double *areas, Matrix2x2 *invJacobi,
    const int3 *edgeBoundaryIDs, double **velocity, double** velocityOld,
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
            if (boundaryID != 1)
                continue;

            const Point2 normalVec = { 1.0, 0.0 };
            const Point2 start = triangleVertices[edge];
            const Point2 end = triangleVertices[(edge + 1) % 3];
            const double halfLength = 0.5 * GEOMETRY::distance(start, end);

            for (int qp = 0; qp < edgeQuadraturePointsNum; ++qp) {
                const Point2 quadraturePoint = edgeQuadraturePoint(start, end, edgeQuadratureFormula[qp].coordinate);
                const Point3 Lcoordinates = GEOMETRY::transformGlobalToLocal(quadraturePoint, cellInvJacobi, triangleVertices[2]);

                aux = simParams.mu * simParams.dt * edgeQuadratureFormula[qp].weight * halfLength;

                for (int i = 0; i < 3; ++i) {
                    const double shapeValueI = *(&Lcoordinates.x + i);

                    for (int j = 0; j < 3; ++j) {
                        const Point2 shapeGradJ = cellInvJacobi * shapeFuncGrad(j);

                        localMatrix[0](i, j) -= aux * shapeValueI * ((4.0 / 3.0) * shapeGradJ.x * normalVec.x + shapeGradJ.y * normalVec.y);
                        localMatrix[1](i, j) -= aux * shapeValueI * (shapeGradJ.x * normalVec.x + (4.0 / 3.0) * shapeGradJ.y * normalVec.y);

                        localRhs[0](i) += aux * shapeValueI * velocity[1][*(&triangle.x + j)] * ((-2.0 / 3.0) * shapeGradJ.y * normalVec.x + shapeGradJ.x * normalVec.y);
                        localRhs[1](i) += aux * shapeValueI * velocity[0][*(&triangle.x + j)] * (shapeGradJ.y * normalVec.x + (-2.0 / 3.0) * shapeGradJ.x * normalVec.y);
                    }
                }
            }
        }

        addLocalToGlobal(triangle, area, localMatrix[0], localRhs[0], rowOffset[0], colIndices[0], matrixValues[0], rhsVector[0]);
        addLocalToGlobal(triangle, area, localMatrix[1], localRhs[1], rowOffset[1], colIndices[1], matrixValues[1], rhsVector[1]);
    }
}

__global__ void kIntegratePressureEquation(int n, const Point2* vertices, const uint3* cells, double* areas, Matrix2x2* invJacobi,
    double** velocityPrediction, const int* rowOffset, const int* colIndices, double* matrixValues, double* rhsVector)
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
    double** velocityPrediction, double* pressure, const int** rowOffset, const int** colIndices, double** matrixValues, double** rhsVector)
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

__global__ void kAccumulatePressureGradient(int n, const uint3* cells, double* areas, Matrix2x2* invJacobi,
    double* pressure, const int* DirichletNodesMap, double *numerator, double *denominator, int component)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint3 triangle = cells[idx];

        if (DirichletNodesMap[triangle.x] == -1 && DirichletNodesMap[triangle.y] == -1 && DirichletNodesMap[triangle.z] == -1)
            return;

        const double area = areas[idx];
        const Matrix2x2 cellInvJacobi = invJacobi[idx];

        Point2 cellGradient = { 0.0, 0.0 };
        for (int i = 0; i < 3; ++i)
            cellGradient += pressure[*(&triangle.x + i)] * shapeFuncGrad(i);

        cellGradient = cellInvJacobi * cellGradient;

        for (int i = 0; i < 3; ++i) {
            const unsigned int nodeI = *(&triangle.x + i);
            if (DirichletNodesMap[nodeI] != -1) {
                atomicAdd(&numerator[DirichletNodesMap[nodeI]], area * *(&cellGradient.x + component));
                atomicAdd(&denominator[DirichletNodesMap[nodeI]], area);
            }
        }
    }
}

__global__ void kFinalizePredictionBC(int n, DirichletNode* targetValues, const DirichletNode* sourceValues, const double* numerator, const double* denominator)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        targetValues[idx].bcValue = sourceValues[idx].bcValue + simParams.dt / simParams.rho * numerator[idx] / denominator[idx];
}

__global__ void kCountBodyEdges(int n, int boundaryID, const uint3* cells, const int3* edgeBoundaryIDs, int* boundaryEdgesCount, int2 *boundaryEdges = nullptr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const int3 cellEdgesBoundaryIDs = edgeBoundaryIDs[idx];

        for(int i = 0; i < 3; ++i)
            if (*(&cellEdgesBoundaryIDs.x + i) == boundaryID){
                int pos = atomicAdd(boundaryEdgesCount, 1);

                if(boundaryEdges){
                    int2 cellEdge;
                    cellEdge.x = idx;
                    cellEdge.y = i;
                    boundaryEdges[pos] = cellEdge;
                }

                return;     //a triangle can not contain 2 boundary edges simultaneously
            }
    }
}

__global__ void kCalculateBodyForces(int n, const Point2 *vertices, const uint3 *cells, Matrix2x2 *invJacobi,
    const int2 *boundaryCells, double **velocity, double* pressure, double4* loadValues)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const int2 cell = boundaryCells[idx];

        const uint3 triangle = cells[cell.x];

        Point2 triangleVertices[3];
        triangleVertices[0] = vertices[triangle.x];
        triangleVertices[1] = vertices[triangle.y];
        triangleVertices[2] = vertices[triangle.z];

        const Matrix2x2 cellInvJacobi = invJacobi[cell.x];

        int edgeVertices[2] = { cell.y, (cell.y + 1) % 3 };

        const Point2 start = triangleVertices[edgeVertices[0]];
        const Point2 end = triangleVertices[edgeVertices[1]];

        const Point2 center = { 0.2, 0.2 };
        const Point2 normal = normalize(0.5 * (start + end) - center);
        const double halfLength = 0.5 * GEOMETRY::distance(start, end);
        const Point2 tangent = { normal.y, -normal.x };

        double4 edgeLoadValues = { 0, 0, 0, 0 };

        for (int qp = 0; qp < edgeQuadraturePointsNum; ++qp) {
            double qPointPressureValue = 0.0;
            double qPointDVtDn = 0.0;

            const Point2 quadraturePoint = edgeQuadraturePoint(start, end, edgeQuadratureFormula[qp].coordinate);
            const Point3 Lcoordinates = GEOMETRY::transformGlobalToLocal(quadraturePoint, cellInvJacobi, triangleVertices[2]);

            for(int i = 0; i < 3; ++i){
                const double shapeValueI = *(&Lcoordinates.x + i);
                qPointPressureValue += pressure[*(&triangle.x + i)] * shapeValueI;

                const Point2 velocityI = { velocity[0][*(&triangle.x + i)], velocity[1][*(&triangle.x + i)] };
                const Point2 shapeGradI = cellInvJacobi * shapeFuncGrad(i);
                qPointDVtDn += dot(velocityI, tangent) * dot(shapeGradI, normal);
            }

            const double weight = edgeQuadratureFormula[qp].weight * halfLength;
            edgeLoadValues.x -= qPointPressureValue * normal.x * weight;
            edgeLoadValues.y -= qPointPressureValue * normal.y * weight;
            edgeLoadValues.z += simParams.mu * qPointDVtDn * tangent.x * weight;
            edgeLoadValues.w += simParams.mu * qPointDVtDn * tangent.y * weight;
        }

        loadValues[idx] = edgeLoadValues;
    }
}

class boundaryLoadsCalculator
{
public:
    boundaryLoadsCalculator(const Mesh2D& mesh_, SimulationParameters &parameters)
        : mesh(mesh_)
        , coeff(2.0 / (parameters.rho * parameters.meanVelocity * parameters.meanVelocity * parameters.thickness))
    {
        allocate_device(&boundaryEdgesCount, 1);
        allocate_device(&totalForces, 1);

        zero_value_device(boundaryEdgesCount, 1);
        blocks = blocksForSize(mesh.getCells().size);
        kCountBodyEdges<<<blocks, gpuThreads>>> (mesh.getCells().size, parameters.bodyBoundaryID, mesh.getCells().data, mesh.getEdgeBoundaryIDs().data, boundaryEdgesCount);

        copy_d2h(boundaryEdgesCount, &hostBoundaryEdgesCount, 1);
        boundaryCells.allocate(hostBoundaryEdgesCount);
        edgeForces.allocate(hostBoundaryEdgesCount);

        zero_value_device(boundaryEdgesCount, 1);
        kCountBodyEdges<<<blocks, gpuThreads>>> (mesh.getCells().size, parameters.bodyBoundaryID, mesh.getCells().data, mesh.getEdgeBoundaryIDs().data, boundaryEdgesCount, boundaryCells.data);

        blocks = blocksForSize(hostBoundaryEdgesCount);

        forcesFile.open("Forces.csv");
        forcesFile << "Time;Cx;Cy" << std::endl;
    }

    ~boundaryLoadsCalculator()
    {
        free_device(boundaryEdgesCount);
        free_device(totalForces);

        if(forcesFile.is_open())
            forcesFile.close();
    }

    void calculateLoads(double time, const deviceVector<double*> &velocity, const deviceVector<double> &pressure)
    {
        edgeForces.clearValues();
        kCalculateBodyForces<<<blocks, gpuThreads>>>(hostBoundaryEdgesCount, mesh.getVertices().data, mesh.getCells().data, mesh.getInvJacobi().data,
            boundaryCells.data, velocity.data, pressure.data, edgeForces.data);

        zero_value_device(totalForces, 1);
        reduceVector<gpuThreads, double, 4><<<1, gpuThreads>>>(hostBoundaryEdgesCount, (double*)edgeForces.data, (double*)totalForces);

        copy_d2h(totalForces, &hostTotalForces, 1);
        double cx, cy;
        cx = (hostTotalForces.x + hostTotalForces.z) * coeff;
        cy = (hostTotalForces.y + hostTotalForces.w) * coeff;
        forcesFile << time << ";" << cx << ";" << cy << std::endl;
    }

private:
    deviceVector<int2> boundaryCells;   //index of the triangle is stored together with the index of the boundary edge
    deviceVector<double4> edgeForces;
    double4 *totalForces;
    double4 hostTotalForces;
    
    int* boundaryEdgesCount;
    int hostBoundaryEdgesCount;
    unsigned int blocks;

    const Mesh2D& mesh;

    const double coeff;

    std::ofstream forcesFile;
};

class VelocityDirichletBCs : public DirichletBCs
{
public:
    VelocityDirichletBCs()
        : DirichletBCs() {};

    void setMesh(const Mesh2D& mesh_) {
        mesh = &mesh_;
        numerator.allocate(DirichletValues.size);
        denominator.allocate(DirichletValues.size);
    }

    void setDirichletValues(const DirichletBCs &VelocityBC, const deviceVector<double> &pressure, int component);

private:
    const Mesh2D *mesh = nullptr;

    deviceVector<double> numerator, denominator;
};

void VelocityDirichletBCs::setDirichletValues(const DirichletBCs& VelocityBC, const deviceVector<double>& pressure, int component)
{
    numerator.clearValues();
    denominator.clearValues();

    unsigned int blocks = blocksForSize(mesh->getCells().size);
    kAccumulatePressureGradient<<<blocks, gpuThreads>>>(mesh->getCells().size, mesh->getCells().data, mesh->getCellArea().data, mesh->getInvJacobi().data,
        pressure.data, nodesToDirichletNodes.data, numerator.data, denominator.data, component);

    blocks = blocksForSize(DirichletValues.size);
    kFinalizePredictionBC<<<blocks, gpuThreads>>> (DirichletValues.size, DirichletValues.data, VelocityBC.getDirichletValues(),
        numerator.data, denominator.data);
}

class CylinderIntegrator : public NumericalIntegrator2D
{
public:
    CylinderIntegrator(const Mesh2D& mesh_)
        : NumericalIntegrator2D(mesh_) { };
    
    const auto &getVelocitySolution() const {
        return velocitySolution;
    }

    auto& getVelocitySolution() {
        return velocitySolution;
    }

    const auto& getVelocitySolutionOld() const {
        return velocitySolutionOld;
    }

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
    deviceVector<double*> velocitySolution;
    deviceVector<double*> velocitySolutionOld;
    deviceVector<double*> velocityPrediction;
    double* pressure;

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

void CylinderIntegrator::setupVelocityPrediction(std::array<SparseMatrixCSR, 2>& csrMatrix, std::array<deviceVector<double>, 2>& rhsVector, const std::array<deviceVector<double>, 2>& velocity)
{
    double* vel[2];
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

void CylinderIntegrator::setupPressure(SparseMatrixCSR& csrMatrix, deviceVector<double>& rhsVector, deviceVector<double>& solution)
{
    pressure = solution.data;
    pressureRhs = rhsVector.data;
    pressureRowOffset = csrMatrix.getRowOffset();
    pressureColIndices = csrMatrix.getColIndices();
    pressureMatrixValues = csrMatrix.getMatrixValues();
}

void CylinderIntegrator::setupVelocityCorrection(std::array<SparseMatrixCSR, 2>& csrMatrix, std::array<deviceVector<double>, 2>& rhsVector,
    const std::array<deviceVector<double>, 2>& velocity, const std::array<deviceVector<double>, 2>& velocityOld)
{
    double* vel[2];
    double* velOld[2];
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

void CylinderIntegrator::assembleVelocityPrediction()
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kIntegrateVelocityPrediction<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getCellArea().data, mesh.getInvJacobi().data,
        mesh.getEdgeBoundaryIDs().data, velocitySolution.data, velocitySolutionOld.data, velocityPredictionRowOffset.data,
        velocityPredictionColIndices.data, velocityPredictionMatrixValues.data, velocityPredictionRhs.data);
}

void CylinderIntegrator::assemblePressureEquation()
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kIntegratePressureEquation<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getCellArea().data, mesh.getInvJacobi().data,
        velocityPrediction.data, pressureRowOffset, pressureColIndices, pressureMatrixValues, pressureRhs);
}

void CylinderIntegrator::assembleVelocityCorrection()
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kIntegrateVelocityCorrection<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getCellArea().data, mesh.getInvJacobi().data,
        velocityPrediction.data, pressure, velocityCorrectionRowOffset.data, velocityCorrectionColIndices.data,
        velocityCorrectionMatrixValues.data, velocityCorrectionRhs.data);
}

int main(int argc, char *argv[]){
	GpuTimer timer;
    ProfilingScope pScope;
    
    pScope.start("Mesh import");

    Mesh2D mesh;
    if(!mesh.loadMeshFromFile("../CylinderMesh.dat"))
        return EXIT_FAILURE;

    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kSetEdgeBoundaryIDs<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getEdgeBoundaryIDs().data);

    pScope.stop();

    pScope.start("Particle seeding");

    ParticleHandler2D particleHandler(&mesh, 2);
    particleHandler.seedParticles();

    pScope.stop();

    const int problemSize = mesh.getVertices().size;

    std::array<DirichletBCs, 2> velocityBCs;
    std::array<VelocityDirichletBCs, 2> velocityPredictionBCs;
    DirichletBCs pressureBCs;
    
    {
        ProfilingScope scope("Boundary conditions setup");

        std::array<std::vector<DirichletNode>, 2> hostVelocityBCs;
        std::vector<DirichletNode> hostPressureBCs;

        const auto& vertices = mesh.getHostVertices();

        hostVelocityBCs[0].reserve(0.1 * vertices.size());
        hostVelocityBCs[1].reserve(0.1 * vertices.size());
        hostPressureBCs.reserve(0.1 * vertices.size());

        for (unsigned i = 0; i < vertices.size(); ++i) {
            const Point2& node = vertices[i];

            if (std::fabs(node.x) < CONSTANTS::DOUBLE_MIN) {
                hostVelocityBCs[0].push_back({ i, 4 * 1.5 * node.y * (h - node.y) / (h * h) });
                hostVelocityBCs[1].push_back({ i, 0.0 });
            } else if (std::fabs(node.x - l) < CONSTANTS::DOUBLE_MIN)
                hostPressureBCs.push_back({ i, 0.0 });
            else if ((std::fabs(node.y) < CONSTANTS::DOUBLE_MIN) || (std::fabs(node.y - h) < CONSTANTS::DOUBLE_MIN)) {
                hostVelocityBCs[0].push_back({i, 0.0});
                hostVelocityBCs[1].push_back({i, 0.0});
            } else if (std::fabs(GEOMETRY::distance(node, cylinderCenter) - 0.05) < 0.001) {
                hostVelocityBCs[0].push_back({ i, 0.0 });
                hostVelocityBCs[1].push_back({ i, 0.0 });
            }
        }

        for (int i = 0; i < 2; ++i) {
            velocityBCs[i].setupDirichletBCs(hostVelocityBCs[i]);
            velocityPredictionBCs[i].setupDirichletBCs(hostVelocityBCs[i]);
            velocityPredictionBCs[i].setMesh(mesh);
            velocityPredictionBCs[i].setupNodeMap(problemSize, hostVelocityBCs[i]);
        }
        pressureBCs.setupDirichletBCs(hostPressureBCs);
    }

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
    hostParams.dt = 0.001;
    hostParams.mu = 0.001;
    hostParams.tFinal = 7.5;
    hostParams.outputFrequency = 100;
    hostParams.exportParticles = 0;
    hostParams.calculateLoads = 1;
    hostParams.bodyBoundaryID = 3;
    hostParams.thickness = 0.1;
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

    CylinderIntegrator integrator(mesh);
    integrator.setupVelocityPrediction(velocityPredictionMatrix, velocityPredictionRhs, velocityPrediction);
    integrator.setupPressure(pressureMatrix, pressureRhs, pressureSolution);
    integrator.setupVelocityCorrection(velocityCorrectionMatrix, velocityCorrectionRhs, velocitySolution, velocitySolutionOld);

    particleHandler.initParticleVelocity(integrator.getVelocitySolution());

    LinearAlgebra LA;
    PreconditionerJacobi precond(problemSize, &LA);
    SolverCG cgSolver(hostParams.tolerance, hostParams.maxIterations, &LA, &precond);
    cgSolver.init(pressureMatrix);

    SolverGMRES gmresSolver(hostParams.tolerance, hostParams.maxIterations, &LA, &precond);
    gmresSolver.init(velocityCorrectionMatrix[0]);

    DataExport dataExport(mesh, &particleHandler);
    dataExport.addScalarDataVector(velocitySolution[0], "velX");
    dataExport.addScalarDataVector(velocitySolution[1], "velY");
    dataExport.addScalarDataVector(velocityPrediction[0], "velPredictionX");
    dataExport.addScalarDataVector(velocityPrediction[1], "velPredictionY");
    dataExport.addScalarDataVector(pressureSolution, "pressure");
    
    dataExport.exportToVTK("solution" + Utilities::intToString(0) + ".vtu");
    if (hostParams.exportParticles)
        dataExport.exportParticlesToVTK("particles" + Utilities::intToString(0) + ".vtu");

    boundaryLoadsCalculator blCalc(mesh, hostParams);

    timer.start();

    //time loop
    unsigned int step_number = 1;
    for (double t = hostParams.dt; t < hostParams.tFinal; t += hostParams.dt, ++step_number) {
        printf("\nTime step no. %u, time = %f\n", step_number, t);
        ProfilingScope stepScope("Simulation step");

        pScope.start("Particle advection");
        particleHandler.advectParticles(integrator.getVelocitySolution(), hostParams.dt, hostParams.particleAdvectionSubsteps);
        pScope.stop();

        pScope.start("Particle velocity projection");
        particleHandler.projectVelocityOntoGrid(integrator.getVelocitySolution());
        pScope.stop();

        for(int i = 0; i < 2; ++i)
            copy_d2d(velocitySolution[i].data, velocitySolutionOld[i].data, problemSize);

        for (int nOuterIter = 0; nOuterIter < 2; ++nOuterIter) {
            //assemble and solve velocity prediction equations
            pScope.start("Velocity prediction");

            pScope.start("Matrix assembly");
            for (int i = 0; i < 2; ++i) {
                velocityPredictionMatrix[i].clearValues();
                velocityPredictionRhs[i].clearValues();
            }
            integrator.assembleVelocityPrediction();
            pScope.stop();
            pScope.start("Linear solver");
            for (int i = 0; i < 2; ++i) {
                velocityPredictionBCs[i].setDirichletValues(velocityBCs[i], pressureSolution, i);
                velocityPredictionBCs[i].applyBCs(velocityPredictionMatrix[i], velocityPredictionRhs[i]);
                gmresSolver.solve(velocityPredictionMatrix[i], velocityPrediction[i], velocityPredictionRhs[i]);
            }
            pScope.stop();
            pScope.stop();

            //assemble and solve the pressure Poisson equation
            pScope.start("Pressure equation");

            pScope.start("Matrix assembly");
            pressureMatrix.clearValues();
            pressureRhs.clearValues();
            integrator.assemblePressureEquation();
            pressureBCs.applyBCs(pressureMatrix, pressureRhs);
            pScope.stop();
            pScope.start("Linear solver");
            cgSolver.solve(pressureMatrix, pressureSolution, pressureRhs);
            pScope.stop();
            pScope.stop();

            //assemble and solve velocity correction equations
            pScope.start("Velocity correction");

            pScope.start("Matrix assembly");
            for (int i = 0; i < 2; ++i) {
                velocityCorrectionMatrix[i].clearValues();
                velocityCorrectionRhs[i].clearValues();
            }
            integrator.assembleVelocityCorrection();
            pScope.stop();
            pScope.start("Linear solver");
            for (int i = 0; i < 2; ++i) {
                velocityBCs[i].applyBCs(velocityCorrectionMatrix[i], velocityCorrectionRhs[i]);
                cgSolver.solve(velocityCorrectionMatrix[i], velocitySolution[i], velocityCorrectionRhs[i]);
            }
            pScope.stop();
            pScope.stop();
        }

        pScope.start("Particle velocity correction");
        particleHandler.correctParticleVelocity(integrator.getVelocitySolution(), integrator.getVelocitySolutionOld());
        pScope.stop();

        if (step_number % hostParams.outputFrequency == 0) {
            ProfilingScope scope("Results output");
            dataExport.exportToVTK("solution" + Utilities::intToString(step_number) + ".vtu");
            if(hostParams.exportParticles)
                dataExport.exportParticlesToVTK("particles" + Utilities::intToString(step_number) + ".vtu");
        }

        pScope.start("Calculation of body loads");
        blCalc.calculateLoads(t, integrator.getVelocitySolution(), pressureSolution);
        pScope.stop();

        float2 times = timer.stop();
        printf("Time of a simulation step: %6.3f ms, total time since start: %6.3f s\n", times.x, 0.001f * times.y);
    }

    return EXIT_SUCCESS;
}
