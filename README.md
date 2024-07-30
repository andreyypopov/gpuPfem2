# gpuFem
CUDA implementation of the finite element method

Solution of the 2D Poisson problem. The whole pipeline is working on GPU:

1. Import of the mesh in DAT format (list of vertices and triangles, can be prepared in SALOME)
2. Setup of the Dirichlet boundary conditions
3. Setup of the CSR matrix structure (analysis of the mesh connectivity)
4. Assembling of the system matrix and right hand side vector by means of numerical integration using Gauss quadrature rules
5. Application of the Dirichlet boundary conditions (changes to both rows and columns corresponding to these boundary degrees of freedom)
6. Solution of the linear system using PCG or GMRES (with or without Jacobi preconditioner)
7. Export of results to a VTK (XML-type) file

**Prerequisites**: CUDA, CMake, OpenMP
