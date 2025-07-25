#include "boundary_loads_calculator_3d.cuh"

#include "../integration/numerical_integrator_3d.cuh"

__global__ void kCountBodyFaces(int n, int boundaryID, const int4* faceBoundaryIDs, int* boundaryFacesCount, int2 *boundaryFaces = nullptr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const int4 tetFacesBoundaryIDs = faceBoundaryIDs[idx];

        for(int i = 0; i < 4; ++i)
            if (*(&tetFacesBoundaryIDs.x + i) == boundaryID){
                const int pos = atomicAdd(boundaryFacesCount, 1);

                if(boundaryFaces){
                    const int bndCell = idx;
                    const int bndFace = i;
                    boundaryFaces[pos] = { bndCell, bndFace };
                }

                return;     //a tetrahedron can not contain 2 boundary faces simultaneously
            }
    }
}

__global__ void kCalculateBoundaryFaceNormals(int n, const Point3 *vertices, const uint4 *cells, const int2 *boundaryFaces, const Point3 &pointInside, Point3 *boundaryFaceNormals)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const auto [bndCell, bndFace] = boundaryFaces[idx];
        const uint4 tet = cells[bndCell];

        Point3 faceVertices[3];
        for(int i = 0; i < 3; ++i)
            faceVertices[i] = vertices[*(&tet.x + (bndFace + i) % 4)];

        const Point3 v1 = faceVertices[1] - faceVertices[0];
        const Point3 v2 = faceVertices[2] - faceVertices[0];
        Point3 normal = normalize(cross(v1, v2));

        const Point3 ov = faceVertices[0] - pointInside;//vector directed from a point inside body towards a vertex of the face
        if (dot(normal, ov) < 0)    //the normal vector should be pointed outwards the body
            normal *= -1.0;

        boundaryFaceNormals[idx] = normal;
    }
}

__global__ void kCalculateBodyForces3D(int n, const uint4 *cells, const GenericMatrix3x3 *invJacobi, const int2 *boundaryFaces,
    const Point3 *boundaryFaceNormals, double **velocity, const double* pressure, double4* loadValues, const GaussPoint2D *quadratureFormula, int quadraturePointsNum, double mu)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const auto [bndCell, bndFace] = boundaryFaces[idx];
        const uint4 tet = cells[bndCell];

        const GenericMatrix3x3 cellInvJacobi = invJacobi[bndCell];

        unsigned int faceVertices[3];
        for (int i = 0; i < 3; ++i)
            faceVertices[i] = *(&tet.x + (bndFace + i) % 4);

        const Point3 normal = boundaryFaceNormals[idx];
        const Point3 tangent = { normal.y, -normal.x, 0.0 };

        double4 edgeLoadValues = { 0, 0, 0, 0 };

        for (int qp = 0; qp < quadraturePointsNum; ++qp) {
            double qPointPressureValue = 0.0;
            double qPointDVtDn = 0.0;

            const Point3 Lcoordinates = quadratureFormula[qp].coordinates;

            for(int i = 0; i < 3; ++i){
                const double shapeValueI = *(&Lcoordinates.x + i);
                qPointPressureValue += pressure[faceVertices[i]] * shapeValueI;

                Point3 velocityI = { velocity[0][faceVertices[i]], velocity[1][faceVertices[i]],velocity[2][faceVertices[i]] };
                const Point3 shapeGradI = cellInvJacobi * shapeFuncGrad3D(i);
                qPointDVtDn += dot(velocityI, tangent) * dot(shapeGradI, normal);
            }

            const double weight = quadratureFormula[qp].weight;
            edgeLoadValues.x -= qPointPressureValue * normal.x * weight;
            edgeLoadValues.y -= qPointPressureValue * normal.y * weight;
            edgeLoadValues.z += mu * qPointDVtDn * tangent.x * weight;
            edgeLoadValues.w += mu * qPointDVtDn * tangent.y * weight;
        }

        loadValues[idx] = edgeLoadValues;
    }
}

BoundaryLoadsCalculator3D::BoundaryLoadsCalculator3D(const Mesh3D& mesh_, const SimulationParameters &parameters_)
    : mesh(mesh_)
    , parameters(parameters_)
    , coeff(2.0 / (parameters.rho * parameters.meanVelocity * parameters.meanVelocity * parameters.thickness * parameters.channelWidth))
{
    allocate_device(&boundaryFacesCount, 1);
    allocate_device(&totalForces, 1);

    zero_value_device(boundaryFacesCount, 1);
    blocks = blocksForSize(mesh.getCells().size);
    kCountBodyFaces<<<blocks, gpuThreads>>> (mesh.getCells().size, parameters.bodyBoundaryID, mesh.getFaceBoundaryIDs().data, boundaryFacesCount);

    copy_d2h(boundaryFacesCount, &hostBoundaryFacesCount, 1);
    boundaryFaces.allocate(hostBoundaryFacesCount);
    boundaryFaceNormals.allocate(hostBoundaryFacesCount);
    faceForces.allocate(hostBoundaryFacesCount);

    zero_value_device(boundaryFacesCount, 1);
    kCountBodyFaces<<<blocks, gpuThreads>>> (mesh.getCells().size, parameters.bodyBoundaryID, mesh.getFaceBoundaryIDs().data, boundaryFacesCount, boundaryFaces.data);

    blocks = blocksForSize(hostBoundaryFacesCount);
    kCalculateBoundaryFaceNormals<<<blocks, gpuThreads>>>(hostBoundaryFacesCount, mesh.getVertices().data, mesh.getCells().data, boundaryFaces.data, parameters.pointInside, boundaryFaceNormals.data);

    forcesFile.open("Forces.csv");
    forcesFile << "Time;Cx;Cy" << std::endl;
}

BoundaryLoadsCalculator3D::~BoundaryLoadsCalculator3D()
{
    free_device(boundaryFacesCount);
    free_device(totalForces);

    if(forcesFile.is_open())
        forcesFile.close();
}

void BoundaryLoadsCalculator3D::calculateLoads(double time, const deviceVector<double *> &velocity,
    const deviceVector<double> &pressure, const GaussPoint2D *faceQuadratureFormula, int faceQuadraturePointsNum)
{
    faceForces.clearValues();
    kCalculateBodyForces3D<<<blocks, gpuThreads>>>(hostBoundaryFacesCount, mesh.getCells().data, mesh.getInvJacobi().data, boundaryFaces.data,
        boundaryFaceNormals.data, velocity.data, pressure.data, faceForces.data, faceQuadratureFormula, faceQuadraturePointsNum, parameters.mu);

    zero_value_device(totalForces, 1);
    reduceVector<gpuThreads, double, 4><<<1, gpuThreads>>>(hostBoundaryFacesCount, (double*)faceForces.data, (double*)totalForces);

    copy_d2h(totalForces, &hostTotalForces, 1);
    const double cx = (hostTotalForces.x + hostTotalForces.z) * coeff;
    const double cy = (hostTotalForces.y + hostTotalForces.w) * coeff;
    forcesFile << time << ";" << cx << ";" << cy << std::endl;
}
