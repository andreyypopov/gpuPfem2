# gpuFem
CUDA implementation of the finite element method

## Algorithm

Solution of the 2D problem using FEM. The whole pipeline is working on GPU:

1. Import of the mesh in DAT format (list of vertices and triangles, can be prepared in SALOME)
2. Setup of the Dirichlet boundary conditions
3. Setup of the CSR matrix structure (analysis of the mesh connectivity)
4. Assembling of the system matrix and right hand side vector by means of numerical integration using Gauss quadrature rules
5. Application of the Dirichlet boundary conditions (changes to both rows and columns corresponding to these boundary degrees of freedom)
6. Solution of the linear system using PCG or GMRES (with or without Jacobi preconditioner)
7. Export of results to a VTK (XML-type) file

## Prerequisites

* C++ compiler (tested on MS VC++ 2022 and g++ 9.4.0)
* OpenMP
* CUDA (version 12; works on 10 as well, if minor changes are made)
* CMake (3.18 or higher)

## Test cases

1. Poisson equation in a rectangular domain
2. Creeping flow (without convective term) in a channel (Poiseuille flow) with partitioned approached (3 equations: velocity prediction equation, Poisson equation for pressure, velocity correction equation).