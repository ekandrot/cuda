#include <iostream>

#define __DRIVER_TYPES_H__
#include "helper_cuda.h"

const int TILE_DIM = 16;
const int totalElements = 4096 * 4096;
const int totalMemorySize = sizeof(float) * totalElements;

cudaEvent_t start, stop;

__global__ void cache_matmul(const float *a, const float *b, float *c) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float t = 0;
    for (int k=0; k<4096; ++k) {
        t += a[k + y*4096] * b[x + k*4096];
    }
    c[x + y*4096] = t;
}


__global__ void shared_matmul(const float *a, const float *b, float *c) {
    __shared__ float sa[16][16];
    __shared__ float sb[16][16];
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float t = 0;
    for (int chunk=0; chunk < gridDim.x; ++chunk) {
        sa[tx][ty] = a[tx+chunk*16 + y*4096];
        sb[tx][ty] = b[x + (ty+chunk*16)*4096];
        __syncthreads();
        for (int k=0; k<16; ++k) {
            t += sa[k][ty] * sb[tx][k];
        }
    }

    c[x + y*4096] = t;
}



void cpu_matmul(const float *a, const float *b, float *c) {
    for (int x=0; x<4096; ++x) {
        for (int y=0; y<4096; ++y) {
            float t = 0;
            for (int k=0; k<4096; ++k) {
                t += a[k + y*4096] * b[x + k*4096];
            }
            c[x+y*4096] = t;
        }
    }
}


int main(int argc, char **argv) {
    checkCudaErrors(cudaSetDevice(0));

    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(totalMemorySize);
    h_b = (float*)malloc(totalMemorySize);
    h_c = (float*)malloc(totalMemorySize);

    float *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc((void**)&d_a, totalMemorySize));
    checkCudaErrors(cudaMalloc((void**)&d_b, totalMemorySize));
    checkCudaErrors(cudaMalloc((void**)&d_c, totalMemorySize));

    for (int i=0; i<totalElements; ++i) {
        h_a[i] = 1;
        h_b[i] = 1;
        h_c[i] = 91;    // dummy value, to make sure we are doing work
    }

    checkCudaErrors(cudaMemcpy(d_a, h_a, totalMemorySize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, totalMemorySize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_c, h_c, totalMemorySize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    dim3 t(TILE_DIM, TILE_DIM, 1);
    dim3 b(4096/TILE_DIM, 4096/TILE_DIM, 1);
    float elapsedTime;

    checkCudaErrors(cudaEventRecord(start, 0));
    cache_matmul<<< b, t >>>(d_a, d_b, d_c);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("cached Elapsed time:  %f ms\n", elapsedTime);

    checkCudaErrors(cudaEventRecord(start, 0));
    shared_matmul<<< b, t >>>(d_a, d_b, d_c);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("shared Elapsed time:  %f ms\n", elapsedTime);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));


    checkCudaErrors(cudaMemcpy(h_c, d_c, totalMemorySize, cudaMemcpyDeviceToHost));
//    cpu_matmul(h_a, h_b, h_c);

    for (int i=0; i<totalElements; ++i) {
        if (h_c[i] != 4096.0) {
            printf("*** First mismatch at %d.  Got %f, was expecting %f  ***\n", i, h_c[i], 4096.0);
            break;
        }
    }

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
    free(h_a);
    free(h_b); 
    free(h_c); 

    return 0;
}
