/*!
 * @file cuda_helper.cuh
 * @brief Helper CUDA functions for different purposes
 * 
 * Includes several constants and the following functions:
 * - count block number;
 * - error handling wrapper;
 * - basic universal kernels (fill a vector with a single value, increase/decrease value of all/specific elements,
 * extract indices of elements corresponding to boolean (unsigned char) mask, etc.);
 * - query amount of free/total GPU memory. 
 */
#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <stdio.h>

const int gpuThreads = 256;             //!< Standard CUDA block size for most part of the kernel functions
const int gpuThreadsMax = 1024;         //!< Increased size of the CUDA block for kernel functions that use shared memory
const int gpuThreads2D = 16;            //!< Decreased size of the CUDA block for kernel functions that use 2D geometry

/*!
 * @brief Calculate the size of CUDA grid for the size of the data to be processed
 * 
 * @param n Number of elements to be processed
 * @param maxThreads Number of threads in a block
 * @return Number of blocks in the grid
 */
inline unsigned int blocksForSize(unsigned int n, unsigned int maxThreads = gpuThreads){
    return (n + maxThreads - 1) / maxThreads;
}

#define checkCusparseErrors(status) {                                                \
    if(status != CUSPARSE_STATUS_SUCCESS){                                          \
        fprintf(stderr, "CUSPARSE API failed at line %d with error: %s (%d)\n",     \
            __LINE__, cusparseGetErrorString(status), status);                      \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
}

#define checkCublasErrors(status) {                                                  \
    if(status != CUBLAS_STATUS_SUCCESS){                                            \
        fprintf(stderr, "CUBLAS API failed at line %d with error: %s (%d)\n",       \
            __LINE__, cublasGetStatusString(status), status);                        \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
}

/*!
 * @brief Function for checking CUDA errors
 * 
 * @tparam T Data type of the result of the function to be checked (cudaError_t)
 * @param result Result of the function execution
 * @param func Function name
 * @param file Name of the source file containg the function
 * @param line Line containing the function
 */
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

/*!
 * @brief Macro for CUDA error checks
 */
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

/*!
 * @brief Macro for asynchronous error checking
 */
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

/*!
 * @brief Function for asynchronous error checking
 * 
 * @param errorMessage Error message
 * @param file Name of the file which caused the error
 * @param line Line of the file which caused the error
 */
inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n", file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__ inline int indexBinarySearch(unsigned int targetElement, const int* elements, int numElements) {
    if (targetElement < elements[0] || targetElement > elements[numElements - 1])
        return -1;

    int leftBorder = 0, rightBorder = numElements - 1;
    if (elements[leftBorder] == targetElement)
        return 0;
    else if (elements[rightBorder] == targetElement)
        return numElements - 1;

    unsigned int middle = (rightBorder - leftBorder) / 2;

    while (rightBorder - leftBorder >= 0) {
        if (elements[middle] == targetElement)
            return middle;
        else if (targetElement < elements[middle])
            rightBorder = middle - 1;
        else
            leftBorder = middle + 1;

        middle = (leftBorder + rightBorder) / 2;
    }

    return -1;
}

#endif // CUDA_HELPER_CUH
