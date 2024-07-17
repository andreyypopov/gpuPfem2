/*!
 * @file cuda_memory.cuh
 * @brief Wrapper functions for memory allocation/deallocation on host and device, memset (device only), copy operations for different configurations
 */
#ifndef CUDA_MEMORY_CUH
#define CUDA_MEMORY_CUH

#include "cuda_helper.cuh"

/*!
 * @brief Allocate a certain amount of device memory in number of items
 * 
 * @tparam T Data type
 * @param ptr Raw pointer to data for allocation
 * @param elements_num Number of elements of the template data type
 */
template<class T>
void allocate_device(T** ptr, size_t elements_num){
    if(elements_num)
        checkCudaErrors(cudaMalloc(ptr, elements_num * sizeof(T)));
    else
        printf("Zero device memory allocation requested\n");
}

/*!
 * @brief Allocate a certain amount of host (pinned) memory in number of items
 * 
 * @tparam T Data type
 * @param ptr Raw pointer to data for allocation
 * @param elements_num Number of elements of the template data type
 */
template<class T>
void allocate_host(T** ptr, size_t elements_num){
    if(elements_num)
        checkCudaErrors(cudaMallocHost(ptr, elements_num * sizeof(T)));
    else
        printf("Zero host memory allocation requested\n");
}

/*!
 * @brief Free device memory
 * 
 * @tparam T Data type
 * @param ptr Raw pointer to data to be deallocated
 */
template<class T>
void free_device(T* ptr){
    if(ptr)
        checkCudaErrors(cudaFree(ptr));
}

/*!
 * @brief Free host memory
 * 
 * @tparam T Data type
 * @param ptr Raw pointer to data to be deallocated
 */
template<class T>
void free_host(T* ptr){
    if(ptr)
        checkCudaErrors(cudaFreeHost(ptr));
}

/*!
 * @brief Set a certain array to zero
 * 
 * @tparam T Data type
 * @param ptr Raw pointer to the data to be set to zero
 * @param elements_num Number of elements of the template data type
 * @param stream CUDA stream (optional)
 */
template<class T>
void zero_value_device(T* ptr, size_t elements_num, cudaStream_t stream = nullptr){
    checkCudaErrors(cudaMemsetAsync(ptr, 0, elements_num * sizeof(T), stream));
}

/*!
 * @brief Copy data from host to device memory
 * 
 * @tparam T Data type
 * @param src Raw pointer in the host memory to copy from
 * @param dst Raw pointer in the device memory to copy to
 * @param elements_num Number of elements of the template data type
 * @param stream CUDA stream (optional)
 */
template<class T>
void copy_h2d(const T* src, const T* dst, size_t elements_num, cudaStream_t stream = nullptr){
    if(elements_num)
        checkCudaErrors(cudaMemcpyAsync((void*)dst, (void*)src, elements_num * sizeof(T), cudaMemcpyHostToDevice, stream));
    else
        printf("Zero host-to-device memory copy requested\n");
}

/*!
 * @brief Copy data from device to host memory
 * 
 * @tparam T Data type
 * @param src Raw pointer in the device memory to copy from
 * @param dst Raw pointer in the host memory to copy to
 * @param elements_num Number of elements of the template data type
 * @param stream CUDA stream (optional)
 */
template<class T>
void copy_d2h(const T* src, const T* dst, size_t elements_num, cudaStream_t stream = nullptr){
    if(elements_num)
        checkCudaErrors(cudaMemcpyAsync((void*)dst, (void*)src, elements_num * sizeof(T), cudaMemcpyDeviceToHost, stream));
    else
        printf("Zero host-to-device memory copy requested\n");
}

/*!
 * @brief Copy data from device to device memory
 * 
 * @tparam T Data type
 * @param src Raw pointer in the device memory to copy from
 * @param dst Raw pointer in the device memory to copy to
 * @param elements_num Number of elements of the template data type
 * @param stream CUDA stream (optional)
 */
template<class T>
void copy_d2d(const T* src, const T* dst, size_t elements_num, cudaStream_t stream = nullptr){
    if(elements_num)
        checkCudaErrors(cudaMemcpyAsync((void*)dst, (void*)src, elements_num * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    else
        printf("Zero device-to-device memory copy requested\n");
}

/*!
 * @brief Copy data from host to constant memory
 * 
 * @tparam T Data type
 * @param src Raw pointer in the host memory to copy from
 * @param dst Raw pointer in the constant memory to copy to
 * @param elements_num Number of elements of the template data type
 * @param stream CUDA stream (optional)
 */
template<class T>
void copy_h2const(const T* src, const void* dst, size_t elements_num, cudaStream_t stream = nullptr){
    if(elements_num)
        checkCudaErrors(cudaMemcpyToSymbolAsync(dst, (void*)src, elements_num * sizeof(T), 0, cudaMemcpyHostToDevice, stream));
    else
        printf("Zero host-to-constant memory copy requested\n");
}

#endif // CUDA_MEMORY_CUH