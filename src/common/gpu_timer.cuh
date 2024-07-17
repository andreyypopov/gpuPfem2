/*!
 * @file gpu_timer.cuh
 * @brief GpuTimer class for measuring time using CUDA events
 */
#ifndef GPU_TIMER_CUH
#define GPU_TIMER_CUH

#include "cuda_helper.cuh"

/*!
 * @brief Class for time measurement using CUDA event mechanism
 * 
 */
class GpuTimer
{
public:
    /*!
     * @brief Construct a new Gpu Timer object
     * 
     * Events are created
     */
    GpuTimer(){
        checkCudaErrors(cudaEventCreate(&startEvent));
        checkCudaErrors(cudaEventCreate(&stopEvent));
    }

    /*!
     * @brief Destroy the Gpu Timer object
     * 
     * Events are destroyed
     */
    ~GpuTimer(){
        if(startEvent)
            checkCudaErrors(cudaEventDestroy(startEvent));
        if(stopEvent)
            checkCudaErrors(cudaEventDestroy(stopEvent));
    }

    /*!
     * @brief Start measuring time
     * 
     * @param stream CUDA stream (optional)
     * 
     * Start event is recorded
     */
    void start(cudaStream_t stream = nullptr){
        if(startEvent)
            checkCudaErrors(cudaEventRecord(startEvent, stream));
    }

    /*!
     * @brief Finish measuring time
     * 
     * @param message Name of the time period/operation (optional)
     * @param stream CUDA stream (optional)
     * @return Measured time in milliseconds
     */
    float stop(const char *message = nullptr, cudaStream_t stream = nullptr){
        if(stopEvent){
            checkCudaErrors(cudaEventRecord(stopEvent, stream));
            checkCudaErrors(cudaEventSynchronize(stopEvent));

            float time;
            checkCudaErrors(cudaEventElapsedTime(&time, startEvent, stopEvent));

            if(message)
                printf("Time for %s: %6.3f ms\n", message, time);

            return time;
        } else
            return 0;
    }

private:
    cudaEvent_t startEvent = nullptr;       //!< Start event
    cudaEvent_t stopEvent = nullptr;        //!< End event
};

#endif // GPU_TIMER_CUH
