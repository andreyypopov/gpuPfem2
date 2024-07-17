# additional functions for setup of the integrator2 library and test applications

macro(setup_cuda target_name)
	enable_language(CUDA)
	find_package(CUDAToolkit)
	include_directories(${CUDAToolkit_INCLUDE_DIRS})
	link_directories(${CUDAToolkit_LIBRARY_DIR})
	set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES "61;70;75")
	set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	target_link_libraries(${target_name} PUBLIC CUDA::cublas CUDA::cusparse)
	add_definitions(-DUSE_CUDA)
endmacro()

macro(setup_openmp target_name)
	find_package(OpenMP)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endmacro()
