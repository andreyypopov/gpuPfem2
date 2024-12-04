CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

include(CMakePrintHelpers)

PROJECT(gpuFemLib LANGUAGES CXX CUDA)
SET(PROJECT_NAME "gpuFemLib")
SET(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if (MSVC)
	SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

INCLUDE(${PROJECT_ROOT_PATH}/cmake/functions.cmake)

SET(PROJECT_SOURCE_PATH ${PROJECT_ROOT_PATH}/src)

SET(src_files
	${PROJECT_SOURCE_PATH}/data_export.cu
	${PROJECT_SOURCE_PATH}/Dirichlet_bcs.cu
	${PROJECT_SOURCE_PATH}/linear_solver.cu
	${PROJECT_SOURCE_PATH}/mesh_2d.cu
	${PROJECT_SOURCE_PATH}/numerical_integrator_2d.cu
	${PROJECT_SOURCE_PATH}/quadrature_formula_1d.cu
	${PROJECT_SOURCE_PATH}/quadrature_formula_2d.cu
	${PROJECT_SOURCE_PATH}/sparse_matrix.cu
	)
SET(src_common_files
	${PROJECT_SOURCE_PATH}/common/cuda_helper.cu
	${PROJECT_SOURCE_PATH}/common/cuda_math.cu
	${PROJECT_SOURCE_PATH}/common/utilities.cpp
)
SET(src_particle_files
	${PROJECT_SOURCE_PATH}/particles/particle_2d.cu
	${PROJECT_SOURCE_PATH}/particles/particle_handler_2d.cu
	)
SET(header_files
	${PROJECT_SOURCE_PATH}/data_export.cuh
	${PROJECT_SOURCE_PATH}/Dirichlet_bcs.cuh
	${PROJECT_SOURCE_PATH}/geometry.cuh
	${PROJECT_SOURCE_PATH}/linear_solver.cuh
	${PROJECT_SOURCE_PATH}/mesh_2d.cuh
	${PROJECT_SOURCE_PATH}/numerical_integrator_2d.cuh
	${PROJECT_SOURCE_PATH}/parameters.cuh
	${PROJECT_SOURCE_PATH}/quadrature_formula_1d.cuh
	${PROJECT_SOURCE_PATH}/quadrature_formula_2d.cuh
	${PROJECT_SOURCE_PATH}/sparse_matrix.cuh
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
	${PROJECT_SOURCE_PATH}/common/profiling.h
	${PROJECT_SOURCE_PATH}/common/utilities.h
	)

source_group("gpuFem" FILES
	${src_files}
	${header_files}
)

source_group("gpuFem/particles" FILES
	${src_particle_files}
	${header_particle_files}
)

source_group("gpuFem/common" FILES
	${src_common_files}
	${header_common_files}
)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${PROJECT_ROOT_PATH}/install" CACHE PATH "Library installation path" FORCE)
endif()

add_library(gpuFemLib SHARED ${src_files} ${src_particle_files} ${src_common_files} ${header_files}
	${header_particle_files} ${header_common_files})

setup_openmp(${PROJECT_NAME})
setup_cuda(${PROJECT_NAME})

install(TARGETS gpuFemLib)
install(FILES ${header_files} DESTINATION include)
install(FILES ${header_common_files} DESTINATION include/common)
install(FILES ${header_particle_files} DESTINATION include/particles)

set_target_properties(gpuFemLib PROPERTIES VERSION 1.0)
set_target_properties(gpuFemLib PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

SET(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Available build configurations" FORCE)
if(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build configuration (available options: ${CMAKE_CONFIGURATION_TYPES})" FORCE)
endif()

MESSAGE("The ${PROJECT_NAME} library will be installed to ${CMAKE_INSTALL_PREFIX}")
cmake_print_variables(CMAKE_BUILD_TYPE)
