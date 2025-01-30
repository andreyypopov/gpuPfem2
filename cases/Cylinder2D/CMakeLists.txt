CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

include(CMakePrintHelpers)

project(Cylinder2D)
SET(PROJECT_NAME "Cylinder2D")
SET(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if (MSVC)
	set_property(DIRECTORY ${PROJECT_ROOT_PATH} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
endif()

INCLUDE(${PROJECT_ROOT_PATH}/../../cmake/functions.cmake)

if(NOT GPUFEMLIB_DIR)
    message(FATAL_ERROR "\"GPUFEMLIB_DIR\" with path to the installed gpuFemLib Library was not set (use the flag -DGPUFEMLIB_DIR=path/to/gpuFemLib).")
endif()

add_executable(${PROJECT_NAME} main.cu)
setup_cuda(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE "${GPUFEMLIB_DIR}/include")
target_link_directories(${PROJECT_NAME} PRIVATE "${GPUFEMLIB_DIR}/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE gpuFemLib)

set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_WITH_INSTALL_RPATH FALSE)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${GPUFEMLIB_DIR}/lib")
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
