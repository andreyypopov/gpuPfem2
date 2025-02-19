CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

include(CMakePrintHelpers)

PROJECT(gpuPfem2Lib LANGUAGES CXX CUDA)
SET(PROJECT_NAME "gpuPfem2Lib")
SET(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if (MSVC)
	SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

INCLUDE(${PROJECT_ROOT_PATH}/cmake/functions.cmake)

SET(PROJECT_SOURCE_PATH ${PROJECT_ROOT_PATH}/src)

SET(src_files
	${PROJECT_SOURCE_PATH}/data_export.cu
	${PROJECT_SOURCE_PATH}/data_export_3d.cu
	${PROJECT_SOURCE_PATH}/Dirichlet_bcs.cu
	${PROJECT_SOURCE_PATH}/mesh_2d.cu
	${PROJECT_SOURCE_PATH}/mesh_3d.cu
	)
SET(src_common_files
	${PROJECT_SOURCE_PATH}/common/cuda_helper.cu
	${PROJECT_SOURCE_PATH}/common/cuda_math.cu
	${PROJECT_SOURCE_PATH}/common/utilities.cpp
)
SET(src_integration_files
	${PROJECT_SOURCE_PATH}/integration/quadrature_formula_1d.cu
	${PROJECT_SOURCE_PATH}/integration/quadrature_formula_2d.cu
	${PROJECT_SOURCE_PATH}/integration/quadrature_formula_3d.cu
)
SET(src_linear_algebra_files
	${PROJECT_SOURCE_PATH}/linear_algebra/linear_solver.cu
	${PROJECT_SOURCE_PATH}/linear_algebra/preconditioners.cu
	${PROJECT_SOURCE_PATH}/linear_algebra/sparse_matrix.cu
	)
SET(src_particle_files
	${PROJECT_SOURCE_PATH}/particles/particle_2d.cu
	${PROJECT_SOURCE_PATH}/particles/particle_handler_2d.cu
	)
SET(header_files
	${PROJECT_SOURCE_PATH}/data_export.cuh
	${PROJECT_SOURCE_PATH}/data_export_3d.cuh
	${PROJECT_SOURCE_PATH}/Dirichlet_bcs.cuh
	${PROJECT_SOURCE_PATH}/geometry.cuh
	${PROJECT_SOURCE_PATH}/mesh_2d.cuh
	${PROJECT_SOURCE_PATH}/mesh_3d.cuh
	${PROJECT_SOURCE_PATH}/parameters.cuh
	)
SET(header_particle_files
	${PROJECT_SOURCE_PATH}/particles/particle_2d.cuh
	${PROJECT_SOURCE_PATH}/particles/particle_handler_2d.cuh
	)
SET(header_common_files
	${PROJECT_SOURCE_PATH}/common/constants.h
	${PROJECT_SOURCE_PATH}/common/cuda_helper.cuh
	${PROJECT_SOURCE_PATH}/common/cuda_math.cuh
	${PROJECT_SOURCE_PATH}/common/cuda_memory.cuh
	${PROJECT_SOURCE_PATH}/common/device_vector.cuh
	${PROJECT_SOURCE_PATH}/common/gpu_timer.cuh
	${PROJECT_SOURCE_PATH}/common/matrix3x3.cuh
	${PROJECT_SOURCE_PATH}/common/matrix4x4.cuh
	${PROJECT_SOURCE_PATH}/common/profiling.h
	${PROJECT_SOURCE_PATH}/common/utilities.h
	)
SET(header_integration_files
	${PROJECT_SOURCE_PATH}/integration/numerical_integrator_2d.cuh
	${PROJECT_SOURCE_PATH}/integration/numerical_integrator_3d.cuh
	${PROJECT_SOURCE_PATH}/integration/quadrature_formula_1d.cuh
	${PROJECT_SOURCE_PATH}/integration/quadrature_formula_2d.cuh
	${PROJECT_SOURCE_PATH}/integration/quadrature_formula_3d.cuh
)
SET(header_linear_algebra_files
	${PROJECT_SOURCE_PATH}/linear_algebra/linear_algebra.h
	${PROJECT_SOURCE_PATH}/linear_algebra/linear_solver.cuh
	${PROJECT_SOURCE_PATH}/linear_algebra/preconditioners.cuh
	${PROJECT_SOURCE_PATH}/linear_algebra/sparse_matrix.cuh
	)

source_group("gpuPfem2" FILES
	${src_files}
	${header_files}
)

source_group("gpuPfem2/particles" FILES
	${src_particle_files}
	${header_particle_files}
)

source_group("gpuPfem2/common" FILES
	${src_common_files}
	${header_common_files}
)

source_group("gpuPfem2/integration" FILES
	${src_integration_files}
	${header_integration_files}
)

source_group("gpuPfem2/linear_algebra" FILES
	${src_linear_algebra_files}
	${header_linear_algebra_files}
)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${PROJECT_ROOT_PATH}/install" CACHE PATH "Library installation path" FORCE)
endif()

add_library(gpuPfem2Lib SHARED ${src_files} ${src_particle_files} ${src_common_files} ${src_integration_files} ${src_linear_algebra_files} ${header_files}
	${header_particle_files} ${header_common_files} ${header_integration_files} ${header_linear_algebra_files})

setup_openmp(${PROJECT_NAME})
setup_cuda(${PROJECT_NAME})

install(TARGETS gpuPfem2Lib)
install(FILES ${header_files} DESTINATION include)
install(FILES ${header_common_files} DESTINATION include/common)
install(FILES ${header_integration_files} DESTINATION include/integration)
install(FILES ${header_linear_algebra_files} DESTINATION include/linear_algebra)
install(FILES ${header_particle_files} DESTINATION include/particles)

set_target_properties(gpuPfem2Lib PROPERTIES VERSION 1.0)
set_target_properties(gpuPfem2Lib PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

SET(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Available build configurations" FORCE)
if(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build configuration (available options: ${CMAKE_CONFIGURATION_TYPES})" FORCE)
endif()

MESSAGE("The ${PROJECT_NAME} library will be installed to ${CMAKE_INSTALL_PREFIX}")
cmake_print_variables(CMAKE_BUILD_TYPE)
