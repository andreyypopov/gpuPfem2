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
    int restartFrequency;           //!< Restart the linear solver cycle each N iterations
    int usePolakRibiereFormula;     //!< Use a different formula for beta coefficient to improve convergence
    int simulationScheme;           //!< Generally refers to Scheme A and Scheme B in pressure accounting

    const char* meshFileName;       //!< File name of the input mesh file
    int outputFrequency;            //!< Output data each N frames
    
    int particleAdvectionSubsteps;  //!< Number of substeps for particle advection within 1 simulation step
    int exportParticles;            //!< Export particles to VTK (boolean flag)

    //loads calculation
    int calculateLoads;             //!< Whether or not loads should be calculated on the body surface
    int bodyBoundaryID;             //!< ID of the boundary edges to be used for forces calculation on the body surface
    double thickness;               //!< Body thickness is the z direction
    double meanVelocity;            //!< Mean flow velocity

    void setDefaultParameters(){
        rho = 1.0;
        mu = 1.0;

        tFinal = 10.0;
        dt = 0.01;
        particleAdvectionSubsteps = 3;

        tolerance = 1e-8;
        maxIterations = 1000;
        restartFrequency = 0;
        usePolakRibiereFormula = 0;
        simulationScheme = 0;

        meshFileName = "";
        outputFrequency = 1;
        exportParticles = 0;

        calculateLoads = 0;
        bodyBoundaryID = -1;
        thickness = 1.0;
        meanVelocity = 1.0;
    }
};

#endif // PARAMETERS_CUH
