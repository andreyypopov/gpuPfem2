#include "cuda_helper.cuh"

__device__ int indexBinarySearch(unsigned int targetElement, const int* elements, int numElements) {
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
