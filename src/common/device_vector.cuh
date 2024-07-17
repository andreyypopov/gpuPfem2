/*!
 * @file device_vector.cuh
 * @brief The deviceVector template class for data storage, allocation and deallocation on GPU
 */
#ifndef DEVICE_VECTOR_CUH
#define DEVICE_VECTOR_CUH

#include "cuda_memory.cuh"
#include "constants.h"

#include <fstream>
#include <vector>

/*!
 * @brief Vector for data stored in the GPU memory
 * 
 * @tparam T Data type
 */
template<class T>
struct deviceVector
{
public:
    T *data = nullptr;      //!< Raw pointer to the array
    int size = 0;           //!< Exact size of the data (can be changed manually)
    int capacity = 0;       //!< Allocated capacity of the array

    /*!
     * @brief Destroy the device Vector object
     * 
     * Memory is freed
     */
    ~deviceVector(){
        free();
    }

    /*!
     * @brief Allocation for a specified number of items
     * 
     * @param new_size Number of items in the vector
     */
    void allocate(int new_size){
        allocate_device(&data, new_size);
        size = new_size;
        capacity = new_size;
    }

    /*!
     * @brief Get the actual size of vector in bytes
     * 
     * @return Size of vector in bytes
     */
    size_t bytes() const {
        return size * sizeof(T);
    }

    /*!
     * @brief Swap the current vector with another one 
     * 
     * @param other Another vector to swap the current with
     * 
     * All properties including raw pointer, size and capacity are swapped
     */
    void swap(deviceVector<T> &other){
        T *tmpData = other.data;
        int tmpSize = other.size;
        int tmpCapacity = other.capacity;

        other.data = this->data;
        other.size = this->size;
        other.capacity = this->capacity;

        this->data = tmpData;
        this->size = tmpSize;
        this->capacity = tmpCapacity;
    }

    /*!
     * @brief Resize the vector to the new size
     * 
     * @param new_size New size in number of elements
     * 
     * If the new size does not exceed the vector capacity, only the size is updated.
     * Otherwise memory reallocation is performed: free the memory (completely, without copying the data),
     * allocate new memory (new size + a certain additional amount), set the size to the requested size.
     */
    void resize(int new_size){
        if(new_size <= capacity)
            size = new_size;
        else {
            free();

            allocate(CONSTANTS::MEMORY_REALLOCATION_COEFFICIENT * new_size);
            size = new_size;

            printf("Reallocation performed\n");
        }
    }

    bool exportToFile(const std::string& filename){
        std::ofstream outputFile(filename.c_str());

        std::vector<T> hostData(size);
        copy_d2h(data, hostData.data(), size);

        if (outputFile.is_open()) {
            for(int i = 0; i < size; ++i)
                outputFile << hostData[i] << std::endl;

            printf("Vector saved to %s\n", filename.c_str());
            return true;
        } else
            return false;
    }

private:
    /*!
     * @brief Free the device memory
     * 
     * If memory was previously allocated, then it is freed and both size and capacity are set to zero.
     */
    void free(){
        if(data){
            free_device(data);
            data = nullptr;
            size = 0;
            capacity = 0;
        }
    }
};

#endif // DEVICE_VECTOR_CUH
