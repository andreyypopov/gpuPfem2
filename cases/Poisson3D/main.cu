#include "data_export_3d.cuh"
#include "mesh_3d.cuh"

#include "common/cuda_math.cuh"
#include "common/gpu_timer.cuh"

#include <vector>

int main(int argc, char *argv[]){
    GpuTimer timer;

    timer.start();

    Mesh3D mesh;
    if(!mesh.loadMeshFromFile("../BoxMesh.dat"))
        return EXIT_FAILURE;

    timer.stop("Mesh import");

    const int problemSize = mesh.getVertices().size;

    DataExport3D dataExport(mesh);
    dataExport.exportToVTK("solution.vtu");

    return EXIT_SUCCESS;
}
