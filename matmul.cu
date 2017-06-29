#include <iostream>
#include "ext_timer.h"
#include "scheduler.h"

#define __DRIVER_TYPES_H__
#include "helper_cuda.h"

const int TD = 32;
const int totalElements = 4096 * 4096;
const int totalMemorySize = sizeof(float) * totalElements;

texture<float,2> texa;
texture<float,2> texb;

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


__global__ void tex_matmul(float *c) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float t = 0;
    for (int k=0; k<4096; ++k) {
        t += tex2D(texa,k,y) * tex2D(texb,x,k);
    }
    c[x + y*4096] = t;
}


__global__ void shared_matmul(const float *a, const float *b, float *c) {
    __shared__ float sa[TD][TD];
    __shared__ float sb[TD][TD];
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float t = 0;
    for (int chunk=0; chunk < gridDim.x; ++chunk) {
        sa[tx][ty] = a[tx+chunk*TD + y*4096];
        sb[tx][ty] = b[x + (ty+chunk*TD)*4096];
        __syncthreads();
        for (int k=0; k<TD; ++k) {
            t += sa[k][ty] * sb[tx][k];
        }
        __syncthreads();
    }

    c[x + y*4096] = t;
}


__global__ void shared_swapped_matmul(const float *a, const float *b, float *c) {
    __shared__ float sa[TD][TD];
    __shared__ float sb[TD][TD];
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float t = 0;
    for (int chunk=0; chunk < gridDim.x; ++chunk) {
        sa[ty][tx] = a[tx+chunk*TD + y*4096];
        sb[ty][tx] = b[x + (ty+chunk*TD)*4096];
        __syncthreads();
        for (int k=0; k<TD; ++k) {
            t += sa[ty][k] * sb[k][tx];
        }
        __syncthreads();
    }

    c[x + y*4096] = t;
}

const size_t MULTI=8;
const size_t TY=(32/MULTI);

__global__ void shared_matmul_unroll(const float *a, const float *b, float *c) {
    __shared__ float sa[TD][TD];
    __shared__ float sb[TD][TD];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = tx + blockIdx.x * TD;
    const int y = ty + blockIdx.y * TD;

    float t[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int chunk=0; chunk < gridDim.x; ++chunk) {
        #pragma unroll
        for (int i=0; i<MULTI; ++i) {
            sa[ty+TY*i][tx] = a[tx+chunk*TD + (y + TY*i)*4096];
            sb[ty+TY*i][tx] = b[x + (ty +chunk*TD + TY*i)*4096];
        }
        __syncthreads();
        #pragma unroll
        for (int k=0; k<TD; ++k) {
            float v = sb[k][tx];
            #pragma unroll
            for(int i=0; i<MULTI; ++i) {
                t[i] += sa[ty+TY*i][k] * v;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0; i<MULTI; ++i) {
        c[x + (y+TY*i)*4096] = t[i];
    }
}



void cpu_matmul(const float *a, const float *b, float *c) {
    for (int x=0; x<4; ++x) {
        for (int y=0; y<4096; ++y) {
            float t = 0;
            for (int k=0; k<4096; ++k) {
                t += a[k + y*4096] * b[x + k*4096];
            }
            c[x+y*4096] = t;
        }
    }
}


struct threaded_matmul : worker {
    float *_a, *_b, *_c;
    threaded_matmul(float *a, float *b, float *c) : _a(a), _b(b), _c(c) {}
    void do_work(int work) {
        int x = work % 4096;
        int y = work / 4096;

        // same code as cached kernel
        float t = 0;
        for (int k=0; k<4096; ++k) {
            t += _a[k + y*4096] * _b[x + k*4096];
        }
        _c[x + y*4096] = t;
    }
};


int main(int argc, char **argv) {
    checkCudaErrors(cudaSetDevice(0));
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(totalMemorySize);
    h_b = (float*)malloc(totalMemorySize);
    h_c = (float*)malloc(totalMemorySize);

    float *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc((void**)&d_a, totalMemorySize));
    checkCudaErrors(cudaMalloc((void**)&d_b, totalMemorySize));
    checkCudaErrors(cudaMalloc((void**)&d_c, totalMemorySize));

    checkCudaErrors( cudaBindTexture2D( NULL, texa, d_a, desc, 4096, 4096, 4096*sizeof(float) ) );
    checkCudaErrors( cudaBindTexture2D( NULL, texb, d_b, desc, 4096, 4096, 4096*sizeof(float) ) );

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

    dim3 t(TD, TD, 1);
    dim3 b(4096/TD, 4096/TD, 1);
    float elapsedTime;

    checkCudaErrors(cudaEventRecord(start, 0));
    tex_matmul<<< b, t >>>(d_c);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("texture Elapsed time:  %f ms\n", elapsedTime);

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

    checkCudaErrors(cudaEventRecord(start, 0));
    shared_swapped_matmul<<< b, t >>>(d_a, d_b, d_c);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("shared swapped Elapsed time:  %f ms\n", elapsedTime);

    dim3 t2(TD, TD/MULTI, 1);
    dim3 b2(4096/TD, 4096/TD, 1);
    checkCudaErrors(cudaEventRecord(start, 0));
    shared_matmul_unroll<<< b2, t2 >>>(d_a, d_b, d_c);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("shared unrolled Elapsed time:  %f ms\n", elapsedTime);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaMemcpy(h_c, d_c, totalMemorySize, cudaMemcpyDeviceToHost));

#if 0   // do CPU - 11 minutes single core, 2 minutes 8-ish cores
    // clear memory again, for the CPU version
    for (int i=0; i<totalElements; ++i) {
        h_a[i] = 1;
        h_b[i] = 1;
        h_c[i] = 93;    // dummy value, to make sure we are doing work
    }
    double startTime = get_wall_time();
    threaded_matmul tmatmul(h_a, h_b, h_c);
    scheduler s(&tmatmul, 4096*4096);
    s.run();
    s.join();
    // 11 minutes of run time...  cpu_matmul(h_a, h_b, h_c);
    double endTime = get_wall_time();
    printf("CPU Elapsed time:  %f sec\n", (float)(endTime-startTime));
#endif

    for (int i=0; i<totalElements; ++i) {
        if (h_c[i] != 4096.0) {
            printf("*** First mismatch at %d.  Got %f, was expecting %f  ***\n", i, h_c[i], 4096.0);
            break;
        }
    }

    cudaUnbindTexture(texa);
    cudaUnbindTexture(texb);

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
    free(h_a);
    free(h_b); 
    free(h_c); 

    return 0;
}

