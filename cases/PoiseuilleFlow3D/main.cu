#include "Dirichlet_bcs.cuh"
#include "mesh_3d.cuh"
#include "parameters.cuh"

#include "common/cuda_math.cuh"
#include "common/gpu_timer.cuh"
#include "common/profiling.h"
#include "common/utilities.h"

#include "integration/numerical_integrator_3d.cuh"
#include "integration/quadrature_formula_2d.cuh"
#include "integration/quadrature_formula_3d.cuh"

#include "linear_algebra/linear_algebra.h"
#include "linear_algebra/linear_solver.cuh"
#include "linear_algebra/preconditioners.cuh"
#include "linear_algebra/sparse_matrix.cuh"

#include "particles/particle_handler_3d.cuh"

#include "postprocessing/data_export_3d.cuh"

#include <vector>

__constant__ GaussPoint3D cellQuadratureFormula[CONSTANTS::MAX_GAUSS_POINTS_3D];
__constant__ int cellQuadraturePointsNum;
__constant__ GaussPoint2D faceQuadratureFormula[CONSTANTS::MAX_GAUSS_POINTS];
__constant__ int faceQuadraturePointsNum;

__constant__ SimulationParameters simParams;

std::string velocityFieldName(int component, bool prediction = false) {
    switch (component) {
    case 0: return (prediction ? "velPredictionX" : "velX");
    case 1: return (prediction ? "velPredictionY" : "velY");
    case 2: return (prediction ? "velPredictionZ" : "velZ");
    default: return std::string();
    }
}

__device__ Point3 normalVector(int boundaryID) {
    switch (boundaryID)
    {
    case 0:
        return { -1.0, 0.0, 0.0 };
    case 1:
        return { 1.0, 0.0, 0.0 };
    case 2:
        return { 0.0, -1.0, 0.0 };
    case 3:
        return { 0.0, 1.0, 0.0 };
    case 4:
        return { 0.0, 0.0, -1.0 };
    case 5:
        return { 0.0, 0.0, 1.0 };
    default:
        return { 0.0, 0.0, 0.0 };
    }
}

__global__ void kSetFaceBoundaryIDs(int n, const Point3 *vertices, const uint4 *cells, int4 *faceBoundaryIDs)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint4 tet = cells[idx];

        Point3 tetVertices[4];
        tetVertices[0] = vertices[tet.x];
        tetVertices[1] = vertices[tet.y];
        tetVertices[2] = vertices[tet.z];
        tetVertices[3] = vertices[tet.w];

        int4 res = { -1, -1, -1, -1 };
        Point3 faceVertices[3];

        for (int i = 0; i < 4; ++i) {
            for (int vert = 0; vert < 3; ++vert)
                faceVertices[vert] = tetVertices[(i + vert) % 4];
            
            const Point3 faceCenter = CONSTANTS::ONE_THIRD * (faceVertices[0] + faceVertices[1] + faceVertices[2]);

            if (std::fabs(faceCenter.x - (-5.0)) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 0;
            else if (std::fabs(faceCenter.x - 5.0) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 1;
            else if (std::fabs(faceCenter.y - (-1.0)) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 2;
            else if (std::fabs(faceCenter.y - 1.0) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 3;
            else if (std::fabs(faceCenter.z) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 4;
            else if (std::fabs(faceCenter.z - 1.0) < CONSTANTS::DOUBLE_MIN)
                *(&res.x + i) = 5;
        }

        faceBoundaryIDs[idx] = res;
    }
}

__global__ void kIntegrateVelocityPrediction(int n, const Point3* vertices, const uint4* cells, double* volumes, GenericMatrix3x3* invJacobi,
    const int4 *faceBoundaryIDs, double **velocity, double** velocityOld,
    const int **rowOffset, const int **colIndices, double **matrixValues, double **rhsVector, double *pressureOld = nullptr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        const uint4 tet = cells[idx];

        const double volume = volumes[idx];

        const GenericMatrix3x3 cellInvJacobi = invJacobi[idx];

        GenericMatrix4x4 localMatrix[3];
        Vector4 localRhs[3];

        double aux;

        //integral over cell
        for (int qp = 0; qp < cellQuadraturePointsNum; ++qp) {
            const Point4 Lcoordinates = cellQuadratureFormula[qp].coordinates;

            for (int i = 0; i < 4; ++i) {
                const Point3 shapeGradI = cellInvJacobi * shapeFuncGrad3D(i);
                const double shapeValueI = *(&Lcoordinates.x + i);

                for (int j = 0; j < 4; ++j) {
                    const Point3 shapeGradJ = cellInvJacobi * shapeFuncGrad3D(j);
                    const double shapeValueJ = *(&Lcoordinates.x + j);

                    aux = simParams.rho * shapeValueI * shapeValueJ * cellQuadratureFormula[qp].weight;

                    for (int k = 0; k < 3; ++k) {
                        localMatrix[k](i, j) += aux;
                        localRhs[k](i) += aux * velocityOld[k][*(&tet.x + j)];
                    }

                    aux = simParams.mu * simParams.dt * cellQuadratureFormula[qp].weight;
                    localMatrix[0](i, j) += aux * (4.0 / 3.0 * shapeGradI.x * shapeGradJ.x + shapeGradI.y * shapeGradJ.y + shapeGradI.z * shapeGradJ.z);
                    localMatrix[1](i, j) += aux * (shapeGradI.x * shapeGradJ.x + 4.0 / 3.0 * shapeGradI.y * shapeGradJ.y + shapeGradI.z * shapeGradJ.z);
                    localMatrix[2](i, j) += aux * (shapeGradI.x * shapeGradJ.x + shapeGradI.y * shapeGradJ.y + 4.0 / 3.0 * shapeGradI.z * shapeGradJ.z);

                    localRhs[0](i) -= aux * ((shapeGradI.y * shapeGradJ.x - 2.0 / 3.0 * shapeGradI.x * shapeGradJ.y) * velocity[1][*(&tet.x + j)] +
                        (shapeGradI.z * shapeGradJ.x - 2.0 / 3.0 * shapeGradI.x * shapeGradJ.z) * velocity[2][*(&tet.x + j)]);
                    localRhs[1](i) -= aux * ((shapeGradI.x * shapeGradJ.y - 2.0 / 3.0 * shapeGradI.y * shapeGradJ.x) * velocity[0][*(&tet.x + j)] +
                        (shapeGradI.z * shapeGradJ.y - 2.0 / 3.0 * shapeGradI.y * shapeGradJ.z) * velocity[2][*(&tet.x + j)]);
                    localRhs[2](i) -= aux * ((shapeGradI.x * shapeGradJ.z - 2.0 / 3.0 * shapeGradI.z * shapeGradJ.x) * velocity[0][*(&tet.x + j)] +
                        (shapeGradI.y * shapeGradJ.z - 2.0 / 3.0 * shapeGradI.z * shapeGradJ.y) * velocity[1][*(&tet.x + j)]);

                    if (pressureOld) {
                        aux = shapeValueI * pressureOld[*(&tet.x + j)] * simParams.dt * cellQuadratureFormula[qp].weight;
                        localRhs[0](i) -= aux * shapeGradJ.x;
                        localRhs[1](i) -= aux * shapeGradJ.y;
                        localRhs[2](i) -= aux * shapeGradJ.z;
                    }
                }
            }
        }

        //integral over boundary faces
        const int4 boundaryIDs = faceBoundaryIDs[idx];
        for (int face = 0; face < 4; ++face) {
            const int boundaryID = *(&boundaryIDs.x + face);
            if (boundaryID != 0 && boundaryID != 1)
                continue;

            const Point3 normalVec = normalVector(boundaryID);

            Point3 tempX{ 0, 0, 0 }, tempY{ 0, 0, 0 }, tempZ{ 0, 0, 0 };
            for (int j = 0; j < 4; ++j) {
                const Point3 shapeGradJ = cellInvJacobi * shapeFuncGrad3D(j);
                const Point3 velocityJ = { velocity[0][*(&tet.x + j)], velocity[1][*(&tet.x + j)], velocity[2][*(&tet.x + j)] };

                tempX.x += (4.0 / 3.0) * shapeGradJ.x * velocityJ.x - (2.0 / 3.0) * shapeGradJ.y * velocityJ.y
                    - (2.0 / 3.0) * shapeGradJ.z * velocityJ.z;
                tempY.y += (-2.0 / 3.0) * shapeGradJ.x * velocityJ.x + (4.0 / 3.0) * shapeGradJ.y * velocityJ.y
                    - (2.0 / 3.0) * shapeGradJ.z * velocityJ.z;
                tempZ.z += (-2.0 / 3.0) * shapeGradJ.x * velocityJ.x - (2.0 / 3.0) * shapeGradJ.y * velocityJ.y
                    + (4.0 / 3.0) * shapeGradJ.z * velocityJ.z;

                aux = shapeGradJ.x * velocityJ.y + shapeGradJ.y * velocityJ.x;
                tempX.y += aux;
                tempY.x += aux;

                aux = shapeGradJ.x * velocityJ.z + shapeGradJ.z * velocityJ.x;
                tempX.z += aux;
                tempZ.x += aux;

                aux = shapeGradJ.y * velocityJ.z + shapeGradJ.z * velocityJ.y;
                tempY.z += aux;
                tempZ.y += aux;
            }

            for (int qp = 0; qp < faceQuadraturePointsNum; ++qp) {
                const Point3 Lcoordinates = faceQuadratureFormula[qp].coordinates;
                
                aux = simParams.mu * simParams.dt * faceQuadratureFormula[qp].weight;
                
                for (int i = 0; i < 3; ++i) {
                    const double shapeValueI = *(&Lcoordinates.x + i);
                    const int indexInTet = (face + i) % 4;
                    localRhs[0](indexInTet) += aux * shapeValueI * dot(tempX, normalVec);
                    localRhs[1](indexInTet) += aux * shapeValueI * dot(tempY, normalVec);
                    localRhs[2](indexInTet) += aux * shapeValueI * dot(tempZ, normalVec);
                }
            }
        }

        for (int k = 0; k < 3; ++k)
            addLocalToGlobal3D(tet, volume, localMatrix[k], localRhs[k], rowOffset[k], colIndices[k], matrixValues[k], rhsVector[k]);
    }
}

__global__ void kIntegratePressureEquation(int n, const Point3* vertices, const uint4* cells, double* volumes, GenericMatrix3x3* invJacobi,
    double** velocityPrediction, const int* rowOffset, const int* colIndices, double* matrixValues, double* rhsVector, double* pressureOld = nullptr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint4 tet = cells[idx];

        const double volume = volumes[idx];
        const GenericMatrix3x3 cellInvJacobi = invJacobi[idx];

        GenericMatrix4x4 localMatrix;
        Vector4 localRhs;

        double aux, aux2;

        //integral over cell
        for (int qp = 0; qp < cellQuadraturePointsNum; ++qp) {
            const Point4 Lcoordinates = cellQuadratureFormula[qp].coordinates;

            for (int i = 0; i < 4; ++i) {
                const Point3 shapeGradI = cellInvJacobi * shapeFuncGrad3D(i);
                const double shapeValueI = *(&Lcoordinates.x + i);

                aux = -simParams.rho / simParams.dt * shapeValueI * cellQuadratureFormula[qp].weight;

                for (int j = 0; j < 4; ++j) {
                    const Point3 shapeGradJ = cellInvJacobi * shapeFuncGrad3D(j);

                    const Point3 velPredictionJ = { velocityPrediction[0][*(&tet.x + j)], velocityPrediction[1][*(&tet.x + j)], velocityPrediction[2][*(&tet.x + j)] };

                    aux2 = dot(shapeGradI, shapeGradJ) * cellQuadratureFormula[qp].weight;
                    localMatrix(i, j) += aux2;
                    if (pressureOld)
                        localRhs(i) += aux2 * pressureOld[*(&tet.x + j)];
                    localRhs(i) += aux * dot(shapeGradJ, velPredictionJ);
                }
            }
        }

        addLocalToGlobal3D(tet, volume, localMatrix, localRhs, rowOffset, colIndices, matrixValues, rhsVector);
    }
}

__global__ void kIntegrateVelocityCorrection(int n, const Point3* vertices, const uint4* cells, double* volumes, GenericMatrix3x3* invJacobi,
    double** velocityPrediction, double* pressure, const int** rowOffset, const int** colIndices, double** matrixValues, double** rhsVector,
    double* pressureOld = nullptr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint4 tet = cells[idx];

        const double volume = volumes[idx];
        const GenericMatrix3x3 cellInvJacobi = invJacobi[idx];

        GenericMatrix4x4 localMatrix[3];
        Vector4 localRhs[3];

        double aux, aux2, pressureValue;

        //integral over cell
        for (int qp = 0; qp < cellQuadraturePointsNum; ++qp) {
            const Point4 Lcoordinates = cellQuadratureFormula[qp].coordinates;

            for (int i = 0; i < 4; ++i) {
                const double shapeValueI = *(&Lcoordinates.x + i);

                aux2 = simParams.dt * shapeValueI * cellQuadratureFormula[qp].weight;

                for (int j = 0; j < 4; ++j) {
                    const Point3 shapeGradJ = cellInvJacobi * shapeFuncGrad3D(j);
                    const double shapeValueJ = *(&Lcoordinates.x + j);

                    aux = simParams.rho * shapeValueI * shapeValueJ * cellQuadratureFormula[qp].weight;
                    pressureValue = pressure[*(&tet.x + j)];
                    if (pressureOld)
                        pressureValue -= pressureOld[*(&tet.x + j)];

                    for (int k = 0; k < 3; ++k) {
                        localMatrix[k](i, j) += aux;
                        localRhs[k](i) += aux * velocityPrediction[k][*(&tet.x + j)] - aux2 * *(&shapeGradJ.x + k) * pressureValue;
                    }
                }
            }
        }

        for (int k = 0; k < 3; ++k)
            addLocalToGlobal3D(tet, volume, localMatrix[k], localRhs[k], rowOffset[k], colIndices[k], matrixValues[k], rhsVector[k]);
    }
}

__global__ void kAccumulatePressureGradient(int n, const uint4* cells, double* volumes, GenericMatrix3x3* invJacobi,
    double* pressure, const int* DirichletNodesMap, double *numerator, double *denominator, int component, double* pressureOld = nullptr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint4 tet = cells[idx];

        if (DirichletNodesMap[tet.x] == -1 && DirichletNodesMap[tet.y] == -1 && DirichletNodesMap[tet.z] == -1 && DirichletNodesMap[tet.w] == -1)
            return;

        const double volume = volumes[idx];
        const GenericMatrix3x3 cellInvJacobi = invJacobi[idx];

        Point3 cellGradient = { 0.0, 0.0, 0.0 };
        for (int i = 0; i < 4; ++i)
            cellGradient += (pressure[*(&tet.x + i)] - (pressureOld ? pressureOld[*(&tet.x + i)] : 0.0)) * shapeFuncGrad3D(i);

        cellGradient = cellInvJacobi * cellGradient;

        for (int i = 0; i < 4; ++i) {
            const unsigned int nodeI = *(&tet.x + i);
            if (DirichletNodesMap[nodeI] != -1) {
                atomicAdd(&numerator[DirichletNodesMap[nodeI]], volume * *(&cellGradient.x + component));
                atomicAdd(&denominator[DirichletNodesMap[nodeI]], volume);
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

class VelocityDirichletBCs : public DirichletBCs
{
public:
    VelocityDirichletBCs()
        : DirichletBCs() {};

    void setMesh(const Mesh3D& mesh_) {
        mesh = &mesh_;
        numerator.allocate(DirichletValues.size);
        denominator.allocate(DirichletValues.size);
    }

    void setDirichletValues(const DirichletBCs &VelocityBC, const deviceVector<double> &pressure, const deviceVector<double> &pressureOld, int component);

private:
    const Mesh3D *mesh = nullptr;

    deviceVector<double> numerator, denominator;
};

void VelocityDirichletBCs::setDirichletValues(const DirichletBCs& VelocityBC, const deviceVector<double>& pressure, const deviceVector<double>& pressureOld, int component)
{
    numerator.clearValues();
    denominator.clearValues();

    unsigned int blocks = blocksForSize(mesh->getCells().size);
    kAccumulatePressureGradient<<<blocks, gpuThreads>>>(mesh->getCells().size, mesh->getCells().data, mesh->getCellVolume().data, mesh->getInvJacobi().data,
        pressure.data, nodesToDirichletNodes.data, numerator.data, denominator.data, component, pressureOld.data);

    blocks = blocksForSize(DirichletValues.size);
    kFinalizePredictionBC<<<blocks, gpuThreads>>> (DirichletValues.size, DirichletValues.data, VelocityBC.getDirichletValues(),
        numerator.data, denominator.data);
}

class PoiseuilleFlowIntegrator : public NumericalIntegrator3D
{
public:
    PoiseuilleFlowIntegrator(const Mesh3D& mesh_)
        : NumericalIntegrator3D(mesh_) { };
    
    const auto &getVelocitySolution() const {
        return velocitySolution;
    }

    auto& getVelocitySolution() {
        return velocitySolution;
    }

    const auto& getVelocitySolutionOld() const {
        return velocitySolutionOld;
    }

    const auto& getVelocityPrediction() const {
        return velocityPrediction;
    }

    //setup pointers (including device ones)
    void setupVelocityPrediction(std::array<SparseMatrixCSR, 3>& csrMatrix, std::array<deviceVector<double>, 3>& rhsVector,
        const std::array<deviceVector<double>, 3>& velocity);

    void setupPressure(SparseMatrixCSR& csrMatrix, deviceVector<double>& rhsVector, deviceVector<double>& solution, deviceVector<double>& solutionOld);

    void setupVelocityCorrection(std::array<SparseMatrixCSR, 3>& csrMatrix, std::array<deviceVector<double>, 3>& rhsVector,
        const std::array<deviceVector<double>, 3>& velocity, const std::array<deviceVector<double>, 3>& velocityOld);

    //assemble matrices and right-hand-side vectors
	void assembleVelocityPrediction();

    void assemblePressureEquation();

    void assembleVelocityCorrection();

private:
    deviceVector<double*> velocitySolution;
    deviceVector<double*> velocitySolutionOld;
    deviceVector<double*> velocityPrediction;
    double* pressure;
    double* pressureOld;

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

void PoiseuilleFlowIntegrator::setupVelocityPrediction(std::array<SparseMatrixCSR, 3>& csrMatrix, std::array<deviceVector<double>, 3>& rhsVector, const std::array<deviceVector<double>, 3>& velocity)
{
    double* vel[3];
    const int* rowOffset[3];
    const int* colIndices[3];
    double* matrixValues[3];
    double* rhs[3];

    for (int i = 0; i < 3; ++i) {
        vel[i] = velocity[i].data;
        rowOffset[i] = csrMatrix[i].getRowOffset();
        colIndices[i] = csrMatrix[i].getColIndices();
        matrixValues[i] = csrMatrix[i].getMatrixValues();
        rhs[i] = rhsVector[i].data;
    }

    velocityPrediction.allocate(3);
    velocityPredictionRhs.allocate(3);
    velocityPredictionRowOffset.allocate(3);
    velocityPredictionColIndices.allocate(3);
    velocityPredictionMatrixValues.allocate(3);

    copy_h2d(vel, velocityPrediction.data, 3);
    copy_h2d(rhs, velocityPredictionRhs.data, 3);
    copy_h2d(rowOffset, velocityPredictionRowOffset.data, 3);
    copy_h2d(colIndices, velocityPredictionColIndices.data, 3);
    copy_h2d(matrixValues, velocityPredictionMatrixValues.data, 3);
}

void PoiseuilleFlowIntegrator::setupPressure(SparseMatrixCSR& csrMatrix, deviceVector<double>& rhsVector, deviceVector<double>& solution, deviceVector<double>& solutionOld)
{
    pressure = solution.data;
    pressureOld = solutionOld.data;
    pressureRhs = rhsVector.data;
    pressureRowOffset = csrMatrix.getRowOffset();
    pressureColIndices = csrMatrix.getColIndices();
    pressureMatrixValues = csrMatrix.getMatrixValues();
}

void PoiseuilleFlowIntegrator::setupVelocityCorrection(std::array<SparseMatrixCSR, 3>& csrMatrix, std::array<deviceVector<double>, 3>& rhsVector,
    const std::array<deviceVector<double>, 3>& velocity, const std::array<deviceVector<double>, 3>& velocityOld)
{
    double* vel[3];
    double* velOld[3];
    const int* rowOffset[3];
    const int* colIndices[3];
    double* matrixValues[3];
    double* rhs[3];

    for (int i = 0; i < 3; ++i) {
        vel[i] = velocity[i].data;
        velOld[i] = velocityOld[i].data;
        rowOffset[i] = csrMatrix[i].getRowOffset();
        colIndices[i] = csrMatrix[i].getColIndices();
        matrixValues[i] = csrMatrix[i].getMatrixValues();
        rhs[i] = rhsVector[i].data;
    }

    velocitySolution.allocate(3);
    velocitySolutionOld.allocate(3);
    velocityCorrectionRhs.allocate(3);
    velocityCorrectionRowOffset.allocate(3);
    velocityCorrectionColIndices.allocate(3);
    velocityCorrectionMatrixValues.allocate(3);

    copy_h2d(vel, velocitySolution.data, 3);
    copy_h2d(velOld, velocitySolutionOld.data, 3);
    copy_h2d(rhs, velocityCorrectionRhs.data, 3);
    copy_h2d(rowOffset, velocityCorrectionRowOffset.data, 3);
    copy_h2d(colIndices, velocityCorrectionColIndices.data, 3);
    copy_h2d(matrixValues, velocityCorrectionMatrixValues.data, 3);
}

void PoiseuilleFlowIntegrator::assembleVelocityPrediction()
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kIntegrateVelocityPrediction<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getCellVolume().data, mesh.getInvJacobi().data,
        mesh.getFaceBoundaryIDs().data, velocitySolution.data, velocitySolutionOld.data, velocityPredictionRowOffset.data,
        velocityPredictionColIndices.data, velocityPredictionMatrixValues.data, velocityPredictionRhs.data, pressureOld);
}

void PoiseuilleFlowIntegrator::assemblePressureEquation()
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kIntegratePressureEquation<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getCellVolume().data, mesh.getInvJacobi().data,
        velocityPrediction.data, pressureRowOffset, pressureColIndices, pressureMatrixValues, pressureRhs, pressureOld);
}

void PoiseuilleFlowIntegrator::assembleVelocityCorrection()
{
    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kIntegrateVelocityCorrection<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getCellVolume().data, mesh.getInvJacobi().data,
        velocityPrediction.data, pressure, velocityCorrectionRowOffset.data, velocityCorrectionColIndices.data,
        velocityCorrectionMatrixValues.data, velocityCorrectionRhs.data, pressureOld);
}

int main(int argc, char *argv[]){
	GpuTimer timer;
    ProfilingScope pScope;
    
    pScope.start("Mesh import");

    Mesh3D mesh;
    if(!mesh.loadMeshFromFile("../ChannelMesh2.dat"))
        return EXIT_FAILURE;

    unsigned int blocks = blocksForSize(mesh.getCells().size);
    kSetFaceBoundaryIDs<<<blocks, gpuThreads>>>(mesh.getCells().size, mesh.getVertices().data, mesh.getCells().data, mesh.getFaceBoundaryIDs().data);

    pScope.stop();

    SimulationParameters hostParams;
    hostParams.setDefaultParameters();
    hostParams.dt = 0.01;
    hostParams.tFinal = 5.0;
    hostParams.simulationScheme = 0;
    hostParams.outputFrequency = 10;
    hostParams.exportParticles = 0;
    hostParams.exportParticleStatistics = 1;
    copy_h2const(&hostParams, &simParams, 1);

    pScope.start("Particle seeding");

    ParticleHandler3D particleHandler(&mesh, hostParams);
    particleHandler.seedParticles();

    pScope.stop();

    const int problemSize = mesh.getVertices().size;

    std::array<DirichletBCs, 3> velocityBCs;
    std::array<VelocityDirichletBCs, 3> velocityPredictionBCs;
    DirichletBCs pressureBCs;
    
    {
        ProfilingScope scope("Boundary conditions setup");

        std::array<std::vector<DirichletNode>, 3> hostVelocityBCs;
        std::vector<DirichletNode> hostPressureBCs;

        const auto& vertices = mesh.getHostVertices();

        hostVelocityBCs[0].reserve(0.1 * vertices.size());
        hostVelocityBCs[1].reserve(0.1 * vertices.size());
        hostVelocityBCs[2].reserve(0.1 * vertices.size());
        hostPressureBCs.reserve(0.1 * vertices.size());

        for (unsigned i = 0; i < vertices.size(); ++i) {
            const Point3& node = vertices[i];

            if (std::fabs(node.x - (-5.0)) < CONSTANTS::DOUBLE_MIN) {
                //option 1 (velocity-driven, constant velocity)
                //hostVelocityBCs[0].push_back({ i, 1.0 });
                //hostVelocityBCs[1].push_back({ i, 0.0 });
                //hostVelocityBCs[2].push_back({ i, 0.0 });
                //option 2 (pressure-driven)
                hostPressureBCs.push_back({ i, 10.0 });
            } else if (std::fabs(node.x - 5.0) < CONSTANTS::DOUBLE_MIN)
                hostPressureBCs.push_back({ i, 0.0 });
            else if ((std::fabs(node.y - (-1.0)) < CONSTANTS::DOUBLE_MIN) || (std::fabs(node.y - 1.0) < CONSTANTS::DOUBLE_MIN) ||
                (std::fabs(node.z) < CONSTANTS::DOUBLE_MIN) || (std::fabs(node.z - 1.0) < CONSTANTS::DOUBLE_MIN)) {
                hostVelocityBCs[0].push_back({i, 0.0});
                hostVelocityBCs[1].push_back({i, 0.0});
                hostVelocityBCs[2].push_back({i, 0.0});
            }
        }

        for (int i = 0; i < 3; ++i) {
            velocityBCs[i].setupDirichletBCs(hostVelocityBCs[i], velocityFieldName(i));
            velocityPredictionBCs[i].setupDirichletBCs(hostVelocityBCs[i], velocityFieldName(i, true));
            velocityPredictionBCs[i].setMesh(mesh);
            velocityPredictionBCs[i].setupNodeMap(problemSize, hostVelocityBCs[i]);
        }
        pressureBCs.setupDirichletBCs(hostPressureBCs, "pressure");
    }

    const auto cellQuadratureGaussPoints = createCellQuadratureFormula(1);
    const auto faceQuadratureGaussPoints = createFaceQuadratureFormula(1);    
    const int cellGaussPointsNum = cellQuadratureGaussPoints.size();
    const int faceGaussPointsNum = faceQuadratureGaussPoints.size();    
    copy_h2const(cellQuadratureGaussPoints.data(), cellQuadratureFormula, cellGaussPointsNum);
    copy_h2const(&cellGaussPointsNum, &cellQuadraturePointsNum, 1);
    copy_h2const(faceQuadratureGaussPoints.data(), faceQuadratureFormula, faceGaussPointsNum);
    copy_h2const(&faceGaussPointsNum, &faceQuadraturePointsNum, 1);

    //matrices, solution and right-hand-side vectors for both component of velocity field (prediction and final ones) and pressure
    std::array<SparseMatrixCSR, 3> velocityCorrectionMatrix;
    std::array<SparseMatrixCSR, 3> velocityPredictionMatrix;
    SparseMatrixCSR pressureMatrix(mesh);

	std::array<deviceVector<double>, 3> velocitySolution;
    std::array<deviceVector<double>, 3> velocitySolutionOld;
	std::array<deviceVector<double>, 3> velocityPrediction;
	deviceVector<double> pressureSolution;
    deviceVector<double> pressureSolutionOld;

    std::array<deviceVector<double>, 3> velocityCorrectionRhs;
    std::array<deviceVector<double>, 3> velocityPredictionRhs;
    deviceVector<double> pressureRhs;
	
	for(int i = 0; i < 3; ++i){
        velocityCorrectionMatrix[i].initialize(mesh);
        velocityPredictionMatrix[i].initialize(mesh);
        velocitySolution[i].allocate(problemSize);
        velocitySolutionOld[i].allocate(problemSize);
		velocityPrediction[i].allocate(problemSize);
        velocityCorrectionRhs[i].allocate(problemSize);
        velocityPredictionRhs[i].allocate(problemSize);
	}
    pressureSolution.allocate(problemSize);
    if (hostParams.simulationScheme == 1)
        pressureSolutionOld.allocate(problemSize);
    pressureRhs.allocate(problemSize);

    //initial solution
    for (int i = 0; i < 3; ++i) {
        velocitySolution[i].clearValues();
        velocityPrediction[i].clearValues();
    }
    pressureSolution.clearValues();
    if (hostParams.simulationScheme == 1)
        pressureSolutionOld.clearValues();

    PoiseuilleFlowIntegrator integrator(mesh);
    integrator.setupVelocityPrediction(velocityPredictionMatrix, velocityPredictionRhs, velocityPrediction);
    integrator.setupPressure(pressureMatrix, pressureRhs, pressureSolution, pressureSolutionOld);
    integrator.setupVelocityCorrection(velocityCorrectionMatrix, velocityCorrectionRhs, velocitySolution, velocitySolutionOld);

    particleHandler.initParticleVelocity(integrator.getVelocitySolution());

    LinearAlgebra LA;

    PreconditionerJacobi JacobiPrecond(problemSize, &LA);
    
    SolverGMRES velocityPredictionSolver(hostParams.tolerance, hostParams.maxIterations, &LA, &JacobiPrecond);
    velocityPredictionSolver.init(velocityPredictionMatrix[0]);

    SolverCG pressureSolver(hostParams.tolerance, hostParams.maxIterations, &LA, &JacobiPrecond);
    pressureSolver.init(pressureMatrix);

    SolverCG velocityCorrectionSolver(hostParams.tolerance, hostParams.maxIterations, &LA, &JacobiPrecond);
    velocityCorrectionSolver.init(velocityCorrectionMatrix[0]);

    DataExport3D dataExport(mesh, &particleHandler);
    dataExport.addVectorDataVector(integrator.getVelocitySolution(), "velocity");
    if (hostParams.exportPredictionVelocity)
        dataExport.addVectorDataVector(integrator.getVelocityPrediction(), "velocityPrediction");
    dataExport.addScalarDataVector(pressureSolution, "pressure");
    
    dataExport.exportToVTK("solution" + Utilities::intToString(0) + ".vtu");
    if (hostParams.exportParticles)
        dataExport.exportParticlesToVTK("particles" + Utilities::intToString(0) + ".vtu");

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

        for(int i = 0; i < 3; ++i)
            copy_d2d(velocitySolution[i].data, velocitySolutionOld[i].data, problemSize);
        if (hostParams.simulationScheme == 1)
            copy_d2d(pressureSolution.data, pressureSolutionOld.data, problemSize);

        for (int nOuterIter = 0; nOuterIter < 1; ++nOuterIter) {
            //assemble and solve velocity prediction equations
            pScope.start("Velocity prediction");

            pScope.start("Matrix assembly");
            for (int i = 0; i < 3; ++i) {
                velocityPredictionMatrix[i].clearValues();
                velocityPredictionRhs[i].clearValues();
            }
            integrator.assembleVelocityPrediction();
            pScope.stop();
            pScope.start("Linear solver");
            for (int i = 0; i < 3; ++i) {
                velocityPredictionBCs[i].setDirichletValues(velocityBCs[i], pressureSolution, pressureSolutionOld, i);
                velocityPredictionBCs[i].applyBCs(velocityPredictionMatrix[i], velocityPredictionRhs[i]);
                velocityPredictionSolver.solve(velocityPredictionMatrix[i], velocityPrediction[i], velocityPredictionRhs[i]);
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
            pressureSolver.solve(pressureMatrix, pressureSolution, pressureRhs);
            pScope.stop();
            pScope.stop();

            //assemble and solve velocity correction equations
            pScope.start("Velocity correction");

            pScope.start("Matrix assembly");
            for (int i = 0; i < 3; ++i) {
                velocityCorrectionMatrix[i].clearValues();
                velocityCorrectionRhs[i].clearValues();
            }
            integrator.assembleVelocityCorrection();
            pScope.stop();
            pScope.start("Linear solver");
            for (int i = 0; i < 3; ++i) {
                velocityBCs[i].applyBCs(velocityCorrectionMatrix[i], velocityCorrectionRhs[i]);
                velocityCorrectionSolver.solve(velocityCorrectionMatrix[i], velocitySolution[i], velocityCorrectionRhs[i]);
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

        float2 times = timer.stop();
        printf("Time of a simulation step: %6.3f ms, total time since start: %6.3f s\n", times.x, 0.001f * times.y);
    }

    return EXIT_SUCCESS;
}
