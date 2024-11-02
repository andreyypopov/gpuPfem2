#ifndef PARAMETERS_CUH
#define PARAMETERS_CUH

#include "common/cuda_memory.cuh"

struct SimulationParameters
{
    double rho;                     //!< Fluid density
    double mu;                      //!< Fluid dynamic viscosity;

    double tFinal;                  //!< Simulation end time
    double dt;                      //!< Simulation time step

    double tolerance;               //!< Linear solver tolerance
    int maxIterations;              //!< Maximum number of iterations of the linear solver

    const char* meshFileName;       //!< File name of the input mesh file
    int outputFrequency;            //!< Output data each N frames
    
    int exportParticles;            //!< Export particles to VTK (boolean flag)

    void setDefaultParameters(){
        rho = 1.0;
        mu = 1.0;

        tFinal = 10.0;
        dt = 0.01;

        tolerance = 1e-8;
        maxIterations = 1000;

        meshFileName = "";
        outputFrequency = 1;
        exportParticles = 0;
    }
};

#endif // PARAMETERS_CUH
