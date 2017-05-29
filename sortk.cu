#include <iostream>

#define __DRIVER_TYPES_H__
#include "helper_cuda.h"

const int K = 10;
const int TD = 1;
const int totalElements = 4096;
const int totalMemorySize = sizeof(float) * totalElements;

/*
*   look for a new home for element i in array a
*   j is the last element of the sorted part of a
*/
__device__ void find_and_insert(float *a, const int i, int j) {
    float t = a[i];
    while (j >= 0 && a[j] > t) {
        a[j+1] = a[j];
        --j;
    }
    a[j+1] = t;
}

__global__ void insertion_sort_k(float *a, const int count, const int k) {
    // sort initial k
    for (int i=1; i<k; ++i) {
        if (a[i-1] > a[i]) {
            find_and_insert(a, i, i-1);
        } 
    }

    // find lowest k uing sorted, plus remainder of array
    float maxv = a[k-1];
    int i = k;
    while (i < count) {
#if 0
        float t = a[i];
        int j = k-1;
        while (j >= 0 && a[j] > t) {
            a[j+1] = a[j];
            --j;
        }
        a[j+1] = t;
#else
        if (a[i] < maxv) {
            find_and_insert(a, i, k-2); // -1 for zero-based, -1 because the last will be gone
            maxv = a[k-1]; // actually 5% faster without this line...
        }
#endif
        ++i;
    }
}


int main(int argc, char **argv) {
    checkCudaErrors(cudaSetDevice(0));

    float *h_a;
    h_a = (float*)malloc(totalMemorySize);

    for (int i=0; i<totalElements; ++i) {
        h_a[i] = 4096-i;
    }
    h_a[totalElements-10] = 8.5;

    float *d_a;
    checkCudaErrors(cudaMalloc((void**)&d_a, totalMemorySize));
    checkCudaErrors(cudaMemcpy(d_a, h_a, totalMemorySize, cudaMemcpyHostToDevice));

    dim3 t(TD, 1, 1);
    dim3 b(1, 1, 1);

    insertion_sort_k<<<b,t>>>(d_a, totalElements, K);

    checkCudaErrors(cudaMemcpy(h_a, d_a, totalMemorySize, cudaMemcpyDeviceToHost));
    for (int i=0; i<K; ++i) {
        std::cout << h_a[i] << ", ";
    }
    std::cout << std::endl;

    checkCudaErrors(cudaFree(d_a));
    free(h_a);

    return 0;
}
