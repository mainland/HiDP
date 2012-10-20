/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../timer.h"

/* Includes, cuda */
#include "cublas.h"

/* Matrix size */
#define M (128)
#define K (2048)
#define N (128)
#define WARPS_PER_BLOCK 8
#define WARP_SIZE 32
#define BLOCK_SIZE (WARPS_PER_BLOCK * WARP_SIZE)

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C)
{
    int i;
    int j;
    int k;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            float prod = 0;
            for (k = 0; k < n; ++k) {
                prod += A[k * n + i] * B[j * n + k];
            }
            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}


template <int blockSize>
__device__ void reduce_block(volatile float *sdata, float mySum)
{
  sdata[threadIdx.x] = mySum;
  __syncthreads();
    // do reduction in shared mem                                                                                                                                                                       
  if (blockSize >= 512) { if (threadIdx.x < 256) { sdata[threadIdx.x] = mySum = mySum+ sdata[threadIdx.x + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (threadIdx.x < 128) { sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (threadIdx.x <  64) { sdata[threadIdx.x] = mySum = mySum +  sdata[threadIdx.x +  64]; } __syncthreads(); }

    if (threadIdx.x < 32)
      {
        if (blockSize >=  64) { sdata[threadIdx.x] = mySum = mySum+ sdata[threadIdx.x + 32]; }
        if (blockSize >=  32) { sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x + 16]; }
        if (blockSize >=  16) { sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x +  8]; }
        if (blockSize >=   8) { sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x +  4]; }
        if (blockSize >=   4) { sdata[threadIdx.x] = mySum = mySum +sdata[threadIdx.x +  2]; }
        if (blockSize >=   2) { sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x +  1]; }
      }
}



__device__ void reduce_warp(volatile float *sdata, float mySum)
{
  int id = threadIdx.x & 0x1F;
  sdata[id ] = mySum;
    // do reduction in shared mem                                                                                                                                               
      {
         sdata[id] = mySum = mySum + sdata[id + 16]; 
         sdata[id] = mySum = mySum + sdata[id +  8];
         sdata[id] = mySum = mySum + sdata[id +  4]; 
         sdata[id] = mySum = mySum +sdata[id +  2]; 
         sdata[id] = mySum = mySum + sdata[id +  1]; 
      }
}


__global__ void matrixMultiply_block(int MM, int NN, int KK, float alpha, float *A, int MMM, float *B, int KKK, float beta, float *C, int NNN)
{
  __shared__ float s_reduce[BLOCK_SIZE];
  int m = blockIdx.x;
  int n = blockIdx.y;
  
  float result = 0.0f;
  A += m;
  B += n * KK;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    {
      result += (*A) * (*B);
      A += MM;
      B++;
    }
  //  s_reduce[thredIdx.x] = result;
  //  __syncthreads();
  reduce_block<BLOCK_SIZE>(&s_reduce[0], result);
  if (threadIdx.x == 0)
    {
      C[m + n * MM] = beta * C[m + n * MM] + alpha * s_reduce[0];
    }
}

__global__ void matrixMultiply_warp(int MM, int NN, int KK, float alpha, float *A, int MMM, float *B, int KKK, float beta, float *C, int NNN)
{
  //  __shared__ float s_reduce[WARPS_PER_BLOCK][WARP_SIZE];
  __shared__ float s_reduce[WARP_SIZE][WARPS_PER_BLOCK];
  int warpId = threadIdx.x/WARP_SIZE;
  int m = blockIdx.x;
  int n = blockIdx.y * warpId;
  
  float result = 0.0f;
  A += m;
  B += n * KK;
 for (int k = threadIdx.x; k < K; k += WARP_SIZE)
    {
      result += (*A) * (*B);
      A += MM;
      B++;
      }
  //  s_reduce[thredIdx.x] = result;
  //  __syncthreads();
   reduce_warp(&(s_reduce[0][warpId]), result);
  if ((threadIdx.x & 0x1F)== 0)
    {
           C[m + n * MM] = beta * C[m + n * MM] + alpha * s_reduce[warpId][0];
    }
}


__global__  void check_result(float *d_residue, float *d_C_ref, float *d_C, int size)
{
  __shared__ float s_reduce[512];
  float residue = 0.0f;
  for (int id = threadIdx.x; id < size; id += blockDim.x)
    residue += (d_C_ref[id] - d_C[id]) * (d_C_ref[id] - d_C[id]);

  reduce_block<512>(&s_reduce[0], residue);
  if (threadIdx.x == 0)
    *d_residue = s_reduce[0];
}


/* Main */
int main(int argc, char** argv)
{    
    cublasStatus status;
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    thrust::device_vector<float> A(M*K);
    thrust::device_vector<float> B(K*N);
    thrust::device_vector<float> C(M*N);

    thrust::device_vector<float> C_block(M*N);
    thrust::device_vector<float> C_warp(M*N);

    /* Clear last error */
    cublasGetError();
    float alpha = 1.0f;
    float beta = 0.0f;
    float *d_A = thrust::raw_pointer_cast(&A[0]);
    float *d_B = thrust::raw_pointer_cast(&B[0]);
    float *d_C = thrust::raw_pointer_cast(&C[0]);
    float *d_C_block = thrust::raw_pointer_cast(&C_block[0]);
    float *d_C_warp = thrust::raw_pointer_cast(&C_warp[0]);

    long flops = M*N*K ;
    /* Performs operation using cublas */
    timer elapsed;
    cublasSgemm('n', 'n', M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }
    float seconds = elapsed.seconds_elapsed();
    printf("matrix multiply cublas M %d N %d K %d gflops %f.\n", M, N, K, (M*N*K) / seconds * 10e-9);


    {
    timer elapsed;
    //    int warps_per_block = 8;
    dim3 blocks(M, N, 1);
    matrixMultiply_block<<<blocks, 256>>>(M, N, K, alpha, d_A, M, d_B, K, beta, d_C_block, N);
    
    float seconds = elapsed.seconds_elapsed();
    printf("matrix multiply block per element M %d N %d K %d gflops %f.\n", M, N, K, (M*N*K) / seconds * 10e-9);
    
    }

    {
    timer elapsed;
    //    int warps_per_block = 8;
    dim3 blocks(M, N/WARPS_PER_BLOCK, 1);
    matrixMultiply_warp<<<blocks, 256>>>(M, N, K, alpha, d_A, M, d_B, K, beta, d_C_warp, N);
    
    float seconds = elapsed.seconds_elapsed();
    printf("matrix multiply warp per element M %d N %d K %d gflops %f.\n", M, N, K, (M*N*K) / seconds * 10e-9);
    
    }

    // checking
    {
    thrust::device_vector<float> residue(1);
    float *d_residue = thrust::raw_pointer_cast(&residue[0]);
    check_result<<<1, 512>>>(d_residue, d_C, d_C_block, N*M);
    thrust::host_vector<float> h_residue = residue;
    printf("diff %f.\n", h_residue[0]);
    }
    {
    thrust::device_vector<float> residue(1);
    float *d_residue = thrust::raw_pointer_cast(&residue[0]);
    check_result<<<1, 512>>>(d_residue, d_C, d_C_warp, N*M);
    thrust::host_vector<float> h_residue = residue;
    printf("diff %f.\n", h_residue[0]);
    }




    
    
#if 0

    float* h_A;
    float* h_B;
    float* h_C;
    float* h_C_ref;
    float* d_A = 0;
    float* d_B = 0;
    float* d_C = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;

    /* Initialize CUBLAS */
    printf("simpleCUBLAS test running..\n");


    /* Allocate host memory for the matrices */
    h_A = (float*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    h_B = (float*)malloc(n2 * sizeof(h_B[0]));
    if (h_B == 0) {
        fprintf (stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }
    h_C = (float*)malloc(n2 * sizeof(h_C[0]));
    if (h_C == 0) {
        fprintf (stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Fill the matrices with test data */
    for (i = 0; i < n2; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;
    }

    /* Allocate device memory for the matrices */
    status = cublasAlloc(n2, sizeof(d_A[0]), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    status = cublasAlloc(n2, sizeof(d_B[0]), (void**)&d_B);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (B)\n");
        return EXIT_FAILURE;
    }
    status = cublasAlloc(n2, sizeof(d_C[0]), (void**)&d_C);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }
    
    /* Performs operation using plain C code */
    simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
    h_C_ref = h_C;

    /* Clear last error */
    cublasGetError();

    /* Performs operation using cublas */
    cublasSgemm('n', 'n', N, N, N, alpha, d_A, N, d_B, N, beta, d_C, N);
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }
    
    /* Allocate host memory for reading back the result from device memory */
    h_C = (float*)malloc(n2 * sizeof(h_C[0]));
    if (h_C == 0) {
        fprintf (stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Read the result back */
    status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;
    for (i = 0; i < n2; ++i) {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }
    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);
    if (fabs(ref_norm) < 1e-7) {
        fprintf (stderr, "!!!! reference norm is 0\n");
        return EXIT_FAILURE;
    }
    printf( "%s\n", (error_norm / ref_norm < 1e-6f) ? "PASSED" : "FAILED");

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    status = cublasFree(d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }
    status = cublasFree(d_B);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }
    status = cublasFree(d_C);
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

    if (argc > 1) {
        if (!strcmp(argv[1], "-noprompt") ||
            !strcmp(argv[1], "-qatest") ) 
        {
            return EXIT_SUCCESS;
        }
    } 
    else
    {
        printf("\nPress ENTER to exit...\n");
        getchar();
    }
#endif
    return EXIT_SUCCESS;
}
