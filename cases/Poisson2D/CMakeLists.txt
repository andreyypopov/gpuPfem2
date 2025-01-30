CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

include(CMakePrintHelpers)

project(Poisson2D)
SET(PROJECT_NAME "Poisson2D")
SET(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if (MSVC)
	set_property(DIRECTORY ${PROJECT_ROOT_PATH} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
endif()

INCLUDE(${PROJECT_ROOT_PATH}/../../cmake/functions.cmake)

if(NOT GPUPFEM2LIB_DIR)
    message(FATAL_ERROR "\"GPUPFEM2LIB_DIR\" with path to the installed gpuPfem2 Library was not set (use the flag -DGPUPFEM2LIB_DIR=path/to/gpuPfem2).")
endif()

add_executable(${PROJECT_NAME} main.cu)
setup_cuda(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE "${GPUPFEM2LIB_DIR}/include")
target_link_directories(${PROJECT_NAME} PRIVATE "${GPUPFEM2LIB_DIR}/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE gpuPfem2Lib)

set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_WITH_INSTALL_RPATH FALSE)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${GPUPFEM2LIB_DIR}/lib")
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
