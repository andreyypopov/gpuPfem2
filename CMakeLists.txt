CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

include(CMakePrintHelpers)

PROJECT(femGpu LANGUAGES CXX CUDA)
SET(PROJECT_NAME "femGpu")
SET(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE(${PROJECT_ROOT_PATH}/cmake/functions.cmake)

SET(PROJECT_SOURCE_PATH ${PROJECT_ROOT_PATH}/src)

SET(src_files
	${PROJECT_SOURCE_PATH}/data_export.cu
	${PROJECT_SOURCE_PATH}/Dirichlet_bcs.cu
	${PROJECT_SOURCE_PATH}/linear_solver.cu
	${PROJECT_SOURCE_PATH}/main.cu
	${PROJECT_SOURCE_PATH}/mesh_2d.cu
	${PROJECT_SOURCE_PATH}/numerical_integrator_2d.cu
	${PROJECT_SOURCE_PATH}/sparse_matrix.cu
	)
SET(src_common_files
	${PROJECT_SOURCE_PATH}/common/cuda_helper.cu
	${PROJECT_SOURCE_PATH}/common/cuda_math.cu
)
SET(header_files
	${PROJECT_SOURCE_PATH}/data_export.cuh
	${PROJECT_SOURCE_PATH}/Dirichlet_bcs.cuh
	${PROJECT_SOURCE_PATH}/linear_solver.cuh
	${PROJECT_SOURCE_PATH}/mesh_2d.cuh
	${PROJECT_SOURCE_PATH}/numerical_integrator_2d.cuh
	${PROJECT_SOURCE_PATH}/sparse_matrix.cuh
	${PROJECT_SOURCE_PATH}/quadrature_formula_2d.cuh
	)
SET(header_common_files
	${PROJECT_SOURCE_PATH}/common/constants.h
	${PROJECT_SOURCE_PATH}/common/cuda_helper.cuh
	${PROJECT_SOURCE_PATH}/common/cuda_math.cuh
	${PROJECT_SOURCE_PATH}/common/cuda_memory.cuh
	${PROJECT_SOURCE_PATH}/common/device_vector.cuh
	${PROJECT_SOURCE_PATH}/common/gpu_timer.cuh
	)

source_group("gpuFem" FILES
	${src_files}
	${header_files}
)

source_group("gpuFem/common" FILES
	${src_common_files}
	${header_common_files}
)

add_executable(femGpu ${src_files} ${src_common_files} ${header_files} ${header_common_files})

setup_openmp(${PROJECT_NAME})
setup_cuda(${PROJECT_NAME})

SET(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Available build configurations" FORCE)
if(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build configuration (available options: ${CMAKE_CONFIGURATION_TYPES})" FORCE)
endif()

cmake_print_variables(CMAKE_BUILD_TYPE)
