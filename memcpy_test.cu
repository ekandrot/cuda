#include <iostream>
#include <cstdlib>
#include <time.h>

#define __DRIVER_TYPES_H__
#include "helper_cuda.h"

const int LOOP = 1;


const size_t TILE_DIM = 32;
const size_t BLOCK_ROWS = 8;
const size_t totalElements = 256 * 1024 * 1024l;
const size_t totalMemorySize = sizeof(float) * totalElements;

cudaEvent_t start, stop;

__global__ void copy1(float *odata, const float *idata) {
    size_t x = blockIdx.x * TILE_DIM + threadIdx.x; // 0..31
    size_t y = blockIdx.y * TILE_DIM + threadIdx.y; // 0..7
    size_t width = gridDim.x * TILE_DIM;

    for (size_t j=0; j<TILE_DIM; j+=BLOCK_ROWS) {
        odata[(y+j)*width + x] = idata[(y+j)*width+x];
    }
}

__global__ void copy2(float *odata, const float *idata) {
    size_t x = blockIdx.x * TILE_DIM + threadIdx.x; // 0..31
    size_t y = blockIdx.y * TILE_DIM + threadIdx.y*4;   // 0,4,8,16,20,24,28
    size_t width = gridDim.x * TILE_DIM;

    for (size_t j=0; j<4; ++j) {
        odata[(y+j)*width + x] = idata[(y+j)*width+x];
    }
}

__global__ void copy3(float *odata, const float *idata) {
    size_t x = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t y = blockIdx.y * TILE_DIM + threadIdx.y;
    size_t width = gridDim.x * TILE_DIM;
    odata[y*width + x] = idata[y*width+x];
}

__global__ void copy3a(float *odata, const float *idata) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // adding this if slows this kernel down to less than copy3,
    // but by a small amount, though makes it work in all memory size cases.
    // it is a trade-off!  If one can guarantee totalElements vs blockDim.x, 
    // then one can get the speed up by removing this check.
    if (tid < totalElements) {
        odata[tid] = idata[tid];
    }
}

__global__ void copy4(float *odata, const float *idata) {
    size_t tid = threadIdx.x + blockDim.x * threadIdx.y;
    tid += blockDim.x * blockDim.y * blockIdx.x;
    size_t width = blockDim.x * blockDim.y * gridDim.x;

    for (size_t j=0; j<totalElements-width; j+=width) {
        odata[tid + j] = idata[tid + j];
    }

    // the final blocks that would overflow
    size_t j=totalElements-width;
    if (tid+j < totalElements) {
        odata[tid + j] = idata[tid + j];
    }
}

float measureCudaMemCpy(float *dev_dst, float *dev_src) {
    float elapsedTimeTotal = 0;
    for (size_t i=0; i<LOOP; ++i) {
        // do a device to device memory copy via kernel, and time it
        checkCudaErrors(cudaEventRecord(start, 0));
        checkCudaErrors(cudaMemcpy(dev_dst, dev_src, totalMemorySize, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));

        float elapsedTime;
        checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
        elapsedTimeTotal += elapsedTime;
    }

    return elapsedTimeTotal / LOOP;
}

float measureCopy1(float *dev_dst, float *dev_src, dim3 BLOCKS) {
    float elapsedTimeTotal = 0;
    for (size_t i=0; i<LOOP; ++i) {
        // do a device to device memory copy via kernel, and time it
        checkCudaErrors(cudaEventRecord(start, 0));
        dim3 THREADS(TILE_DIM, BLOCK_ROWS, 1);
        copy1 <<< BLOCKS, THREADS >>>(dev_dst, dev_src);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));

        float elapsedTime;
        checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
        elapsedTimeTotal += elapsedTime;
    }

    return elapsedTimeTotal / LOOP;
}

float measureCopy2(float *dev_dst, float *dev_src, dim3 BLOCKS) {
    float elapsedTimeTotal = 0;
    for (size_t i=0; i<LOOP; ++i) {
        // do a device to device memory copy via kernel, and time it
        checkCudaErrors(cudaEventRecord(start, 0));
        dim3 THREADS(TILE_DIM, BLOCK_ROWS, 1);
        copy2 <<< BLOCKS, THREADS >>>(dev_dst, dev_src);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));

        float elapsedTime;
        checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
        elapsedTimeTotal += elapsedTime;
    }

    return elapsedTimeTotal / LOOP;
}

float measureCopy3(float *dev_dst, float *dev_src, dim3 BLOCKS) {
    float elapsedTimeTotal = 0;
    for (size_t i=0; i<LOOP; ++i) {
        // do a device to device memory copy via kernel, and time it
        checkCudaErrors(cudaEventRecord(start, 0));
        dim3 THREADS(TILE_DIM, TILE_DIM, 1);
        copy3 <<< BLOCKS, THREADS >>>(dev_dst, dev_src);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));

        float elapsedTime;
        checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
        elapsedTimeTotal += elapsedTime;
    }

    return elapsedTimeTotal / LOOP;
}

/*
    changed this to use occupancy API.  it calculated the same numbers I had handcoded,
    but this makes it past/future proof.  the copy3a kernel with this code to drive makes
    the least assumptions about the hardware, and is the fastest.

    https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    was the article that covered the use of the occupancy api, and the code I copied for debug printing.
*/
float measureCopy3a(float *dev_dst, float *dev_src) {
    int blockSize=0; // to supress warning, assign to zero
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, copy3a, 0, 0);
    gridSize = (totalElements + blockSize - 1) / blockSize;

    float elapsedTimeTotal = 0;
    for (int i=0; i<LOOP; ++i) {
        // do a device to device memory copy via kernel, and time it
        checkCudaErrors(cudaEventRecord(start, 0));
        copy3a <<< gridSize, blockSize >>>(dev_dst, dev_src);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));

        float elapsedTime;
        checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
        elapsedTimeTotal += elapsedTime;
    }

    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 copy3a, blockSize, 
                                                 0);
    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize);
    //printf("   [Debugging Info for copy3a] Launched blocks of size %d. Theoretical occupancy: %f\n", blockSize, occupancy);

    return elapsedTimeTotal / LOOP;
}

float measureCopy4(float *dev_dst, float *dev_src, dim3 BLOCKS) {
    float elapsedTimeTotal = 0;
    for (size_t i=0; i<LOOP; ++i) {
        // do a device to device memory copy via kernel, and time it
        checkCudaErrors(cudaEventRecord(start, 0));
        dim3 THREADS(TILE_DIM, TILE_DIM, 1);
        copy4 <<< BLOCKS, THREADS >>>(dev_dst, dev_src);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));

        float elapsedTime;
        checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
        elapsedTimeTotal += elapsedTime;
    }

    return elapsedTimeTotal / LOOP;
}

int main(int argc, char **argv) {
    srand(time(NULL));
    checkCudaErrors(cudaSetDevice(0));

    float *host_src, *host_dst;
    host_src = (float*)malloc(totalMemorySize);
    host_dst = (float*)malloc(totalMemorySize);

    float *dev_src, *dev_dst;
    checkCudaErrors(cudaMalloc((void**)&dev_src, totalMemorySize));
    checkCudaErrors(cudaMalloc((void**)&dev_dst, totalMemorySize));

    // fill memory on host side
    float theRand = rand();
    for (size_t i=0; i<totalElements; ++i) {
        host_src[i] = theRand;
    }

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(dev_src, host_src, totalMemorySize, cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    float timePerLoop;


    timePerLoop = measureCudaMemCpy(dev_dst, dev_src);
    printf("mem Elapsed time:  %f s\n", timePerLoop / 1000.0);

    timePerLoop = measureCopy1(dev_dst, dev_src, dim3(totalElements/1024/32/32, 1024, 1));
    printf("1 Elapsed time:  %f s\n", timePerLoop / 1000.0);

    timePerLoop = measureCopy1(dev_dst, dev_src, dim3(totalElements/1024/32/32/32, 1024*32, 1));
    printf("1 Elapsed time:  %f s\n", timePerLoop / 1000.0);

    timePerLoop = measureCopy2(dev_dst, dev_src, dim3(totalElements/1024, 1024/32/32, 1));
    printf("2 Elapsed time:  %f s\n", timePerLoop / 1000.0);

    timePerLoop = measureCopy2(dev_dst, dev_src, dim3(totalElements/1024/32/32, 1024, 1));
    printf("2 Elapsed time:  %f s\n", timePerLoop / 1000.0);

    timePerLoop = measureCopy2(dev_dst, dev_src, dim3(totalElements/1024/32/32/32, 1024*32, 1));
    printf("2 Elapsed time:  %f s\n", timePerLoop / 1000.0);

//    timePerLoop = measureCopy2(dev_dst, dev_src, dim3(totalElements/1024/32/32/64, 1024*64, 1));
//    printf("Elapsed time:  %f s\n", timePerLoop / 1000.0);

    timePerLoop = measureCopy3(dev_dst, dev_src, dim3(totalElements/1024/32/32/32, 1024*32, 1));
    printf("3 Elapsed time:  %f s\n", timePerLoop / 1000.0);

    timePerLoop = measureCopy3a(dev_dst, dev_src);
    printf("3a Elapsed time:  %f s\n", timePerLoop / 1000.0);

    timePerLoop = measureCopy4(dev_dst, dev_src, dim3(26, 1, 1));
    printf("4 Elapsed time:  %f s\n", timePerLoop / 1000.0);

    timePerLoop = measureCopy4(dev_dst, dev_src, dim3(1024, 1, 1));
    printf("4 Elapsed time:  %f s\n", timePerLoop / 1000.0);


    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaMemcpy(host_dst, dev_dst, totalMemorySize, cudaMemcpyDeviceToHost));
    for (size_t i=0; i<totalElements; ++i) {
        if (host_dst[i] != host_src[i]) {
            printf("*** First mismatch at %ld.  Got %f, was expecting %f  ***\n", i, host_dst[i], host_src[i]);
            break;
        }
    }

    checkCudaErrors(cudaFree(dev_src));
    checkCudaErrors(cudaFree(dev_dst));
    free(host_src);
    free(host_dst); 

    return 0;
}

