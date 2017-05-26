// Copied from the question at:
// https://stackoverflow.com/questions/5551020/cublas-matrix-multiplication
// then modified so I could compare to hand-coded timings.

#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"
#define HA 4096
#define WA 4096
#define WB 4096
#define HB WA 
#define WC WB   
#define HC HA  
#define index(i,j,ld) (((j)*(ld))+(i))

#define __DRIVER_TYPES_H__
#include "helper_cuda.h"


cudaEvent_t start, stop;


void printMat(float*P,int uWP,int uHP){
//printf("\n %f",P[1]);
int i,j;
for(i=0;i<uHP;i++){

    printf("\n");

    for(j=0;j<uWP;j++)
        printf("%f ",P[index(i,j,uHP)]);
        //printf("%f ",P[i*uWP+j]);
}
}




 int  main (int argc, char** argv) {
    cublasStatus status;
        int i,j;
        cublasInit();

        float *A = (float*)malloc(HA*WA*sizeof(float));
        float *B = (float*)malloc(HB*WB*sizeof(float));
        float *C = (float*)malloc(HC*WC*sizeof(float));
    if (A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    if (B == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    if (C == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
        }


    for (i=0;i<HA;i++)
        for (j=0;j<WA;j++)
            A[index(i,j,HA)] = (float) 1;//index(i,j,HA);   
    for (i=0;i<HB;i++)
        for (j=0;j<WB;j++)
            B[index(i,j,HB)] = (float) 1;//index(i,j,HB); 
    /*
    for (i=0;i<HA*WA;i++)
    A[i]=(float) i;
    for (i=0;i<HB*WB;i++)
    B[i]=(float) i;         */  


        float* AA; float* BB; float* CC;

    /*ALLOCATE ON THE DEVICE*/
    status=cublasAlloc(HA*WA,sizeof(float),(void**)&AA);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
        return EXIT_FAILURE;
        }

        status=cublasAlloc(HB*WB,sizeof(float),(void**)&BB);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
        return EXIT_FAILURE;
        }

        status=cublasAlloc(HC*WC,sizeof(float),(void**)&CC);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
        return EXIT_FAILURE;
        }

    /*SET MATRIX*/
        status=cublasSetMatrix(HA,WA,sizeof(float),A,HA,AA,HA);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
        return EXIT_FAILURE;
        }

        status=cublasSetMatrix(HB,WB,sizeof(float),B,HB,BB,HB);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
        return EXIT_FAILURE;
        }

    /*KERNEL*/
    float elapsedTime;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));
        cublasSgemm('n','n',HA,WB,WA,1,AA,HA,BB,HB,0,CC,HC);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("cublas Elapsed time:  %f ms\n", elapsedTime);

        status = cublasGetError();
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
        }
        cublasGetMatrix(HC,WC,sizeof(float),CC,HC,C,HC);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device read error (A)\n");
        return EXIT_FAILURE;
        }


    /* PERFORMANCE OUTPUT*/

/*
    printf("\nMatriz A:\n");
    printMat(A,WA,HA);
    printf("\nMatriz B:\n");
    printMat(B,WB,HB);
    printf("\nMatriz C:\n");
    printMat(C,WC,HC);
*/

    for (int j=0; j<4096; ++j) {
        for (int i=0; i<4096; ++i) {
            if (C[index(i,j,HC)] != 4096.0) {
                printf("Failed at (%d,%d):  %f\n", i, j, C[index(i,j,HC)]);
                goto cleanup;
            }
        }
    }
cleanup:
        free( A );  free( B );  free ( C );
        status = cublasFree(AA);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
        }
        status = cublasFree(BB);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
        }
        status = cublasFree(CC);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
    }

        /* Shutdown */
        status = cublasShutdown();
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
        }


    return EXIT_SUCCESS; 
}

