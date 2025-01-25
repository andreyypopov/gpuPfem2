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

## Prerequisites

* C++ compiler (tested on MS VC++ 2022 and g++ 9.4.0)
* OpenMP
* CUDA (version 12; works on 10 as well, if minor changes are made)
* CMake (3.18 or higher)

## Test cases

1. Poisson equation in a rectangular domain
2. Creeping flow (without convective term) in a channel (Poiseuille flow) with partitioned approached (3 equations: velocity prediction equation, Poisson equation for pressure, velocity correction equation).
3. Poiseuille flow in a channel
4. Flow past a cylinder in a channel (test 2D-2 from _Schäfer M., Turek S., Durst F., Krause E., Rannacher R. (1996). Benchmark Computations of Laminar Flow Around a Cylinder_)
