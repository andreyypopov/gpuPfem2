# gpuPfem2

CUDA implementation of the particle finite element method, 2<sup>nd</sup> generation

## Algorithm

Solution of the 2D problem using PFEM-2. The whole pipeline, including both particle and mesh steps, is working on GPU:

1. Import of the mesh in DAT format (list of vertices and triangles, can be prepared in SALOME)
2. Setup of the Dirichlet boundary conditions
3. Setup of the CSR matrix structure (analysis of the mesh connectivity)
4. Initial seeding of the particles
5. Particle advection, including sorting against mesh cells and checking particle distribution
6. Projection of velocity from particles onto grid
7. Assembling of the system matrix and right hand side vector by means of numerical integration using Gauss quadrature rules
8. Application of the Dirichlet boundary conditions (changes to both rows and columns corresponding to these boundary degrees of freedom)
9. Solution of the linear system using PCG or GMRES with or without a preconditioner (Jacobi, Incomplete LU/Cholesky decomposition)
10. Correction of particle velocity
11. Export of results to a VTK (XML-type) file
12. Calculation of aerodynamic loads and coefficients (drag and lift force) on bodies

Partial support of solution in 3D - only FEM on a tetrahedral Eulerian mesh, without particles.

## Prerequisites

* C++ compiler (tested on MS VC++ 2022 and g++ 9.4.0)
* OpenMP
* CUDA (version 12; works on 10 as well, if minor changes are made)
* CMake (3.18 or higher)

## Building the library and running test cases

The library is built in a conventional way, e.g. (if executed from the path `./build`)

```
cmake ..
make
make install
```

CMake options:

1. `CMAKE_INSTALL_PREFIX` - installation directory. If it is not set, it is defaulted to `install` directory.
2. `CMAKE_BUILD_TYPE` - build configuration. If it is not set, it is defaulted to RelWithDebInfo.

To build a test case path to the installation directory should be provided:

```
cmake -DGPUPFEM2LIB_DIR=/path/to/library .
make
```

## Test cases

1. Poisson equation in a rectangular domain
2. Creeping flow (without convective term) in a channel (Poiseuille flow) with partitioned approached (3 equations: velocity prediction equation, Poisson equation for pressure, velocity correction equation).
3. Poiseuille flow in a channel
4. Flow past a cylinder in a channel (test 2D-2 from _Schäfer M., Turek S., Durst F., Krause E., Rannacher R. (1996). Benchmark Computations of Laminar Flow Around a Cylinder_)
5. Poisson equation in a 3D box

## Governing equations and splitting schemes

### Incompressible viscous fluid flow

Navier - Stokes equations and incompressibility equation are solved:
```math
\begin{gather*}
\rho\left(\frac{\partial\boldsymbol{V}}{\partial t} + (\boldsymbol{V}\cdot\nabla)\boldsymbol{V}\right) = -\nabla p + \nabla\cdot\hat{\tau} + \rho\boldsymbol{g},\\
\nabla\cdot\boldsymbol{V}=0,
\end{gather*}
```
where $`\hat{\tau}`$ is the deviatoric stress tensor with following components ($`\mu`$ is the dynamic viscosity coefficient):
```math
\tau_{ij} = \mu\left(\frac{\partial V_i}{\partial x_j} + \frac{\partial V_j}{\partial x_i} - \frac23\delta_{ij}\frac{\partial V_k}{\partial x_k} \right).
```

Particles are used to simulate advection, whereas the reduced system (without convective term) is solved on the Eulerian mesh using traditional FEM:
```math
\begin{gather*}
\rho\frac{\partial\boldsymbol{V}}{\partial t} = -\nabla p + \nabla\cdot\hat{\tau} + \rho\boldsymbol{g},\\
\nabla\cdot\boldsymbol{V}=0.
\end{gather*}
```

### Splitting schemes

Rather than using a monolithic scheme, we adopt a fractional step approach to decouple velocity from pressure. Solution is obtained in 3 steps, which are a bit different depending on how pressure is taken into account (Schemes A and B in _Zienkiwicz O., Taylor R. (2000). The Finite Element Method, Vol. 3: Fluid Dynamics_). Following equations are solved ($`\boldsymbol{V}^{n+1/2}`$ is the velocity prediction field):

1. Velocity prediction equation

    1.1. Scheme A: $`\rho\dfrac{\boldsymbol{V}^{n+1/2} - \boldsymbol{V}^n}{\Delta t} = \nabla\cdot\hat{\tau}^* + \rho\boldsymbol{g}`$

    1.2. Scheme B: $`\rho\dfrac{\boldsymbol{V}^{n+1/2} - \boldsymbol{V}^n}{\Delta t} = -\nabla p^{n} + \nabla\cdot\hat{\tau}^{n+1/2} + \rho\boldsymbol{g}`$

    In $`\hat{\tau}^{n+1/2}`$ component of velocity field corresponding to the equation is approximated implicitly (e.g., $`V_x`$ for the $`x`$ equation), while the other one is approximated explicitly using its known values.
2. Poisson pressure equation

    2.1. Scheme A: $`\Delta p^{n+1} = \dfrac{\rho}{\Delta t}\nabla\cdot\boldsymbol{V}^{n+1/2}`$

    2.2. Scheme B: $`\Delta p^{n+1} = \Delta p^n + \dfrac{\rho}{\Delta t}\nabla\cdot\boldsymbol{V}^{n+1/2}`$
3. Velocity correction equation

    3.1. Scheme A: $`\rho\dfrac{\boldsymbol{V}^{n+1} - \boldsymbol{V}^{n+1/2}}{\Delta t} = -\nabla p^{n+1}`$

    3.2. Scheme B: $`\rho\dfrac{\boldsymbol{V}^{n+1} - \boldsymbol{V}^{n+1/2}}{\Delta t} = -(\nabla p^{n+1} - \nabla p^n)`$

    Actually, it is just an algebraic dependence, not an equation, but as both velocity and pressure values at stores at the same positions (mesh nodes), it is not easy to compute the pressure gradient in a straightforward way. Therefore it is solved using FEM.

Schemes A and B were implemented in the Cylinder2D and PoiseuilleFlow2D cases. The switch is performed using `simulationScheme` parameter in the `SimulationParameters` structure (0 corresponds to Scheme A, 1 - to Scheme B).