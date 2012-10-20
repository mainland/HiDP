
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include "cublas.h"
#include "matrixMul.h"

// A : M*K  B : K*N   C M*N
/*#define fetchA(i,j) (*(A+KK*i + j))
#define fetchB(i,j) (*(B+NN*i + j))
#define fetchC(i,j) (*(C+NN*i + j))*/
#define fetchA(i,j) (*(A+(i) + (j)*MM))
#define fetchB(i,j) (*(B+(i) + (j)*KK))
#define fetchC(i,j) (*(C+(i) + (j)*MM))


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
  sdata[id] = mySum;
  // do reduction in shared mem 
  if (id < 16)
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
  //  A += m*KK;
  //  B += n;
  for (int k = threadIdx.x; k < KK; k += blockDim.x)
      result += fetchA(m,k) * fetchB(k,n);
  //  s_reduce[thredIdx.x] = result;
  //  __syncthreads();
  reduce_block<BLOCK_SIZE>(&s_reduce[0], result);
  if (threadIdx.x == 0)
    {
      fetchC(m,n) = beta * fetchC(m,n) + alpha * s_reduce[0];
    }
}



__global__ void matrixMultiply_warp(int MM, int NN, int KK, float alpha, float *A, int MMM, float *B, int KKK, float beta, float *C, int NNN)
{
  //  printf("here0.\n");
    __shared__ float s_reduce[WARPS_PER_BLOCK][WARP_SIZE];
    //  __shared__ float s_reduce[WARP_SIZE][WARPS_PER_BLOCK];
  int warpId = threadIdx.x/WARP_SIZE;
  int warpIndex = threadIdx.x & 0x1F;
  int m = blockIdx.x;
  int n = blockIdx.y * WARPS_PER_BLOCK +  warpId;
  
  if (n >= NN || m >= MM) return;

  float result = 0.0f;
  //  A += m;
  //  B += n * KK;
 for (int k = warpIndex; k < KK; k += WARP_SIZE)
    {
      /*      if (k<2)
              printf("multipplying %f %f in block %d %d.\n", fetchA(m,k), fetchB(k,n), m, n);*/
      
      result += fetchA(m,k) * fetchB(k,n);
      //      result += (*A) * (*B);
      //p      A += MM;
      //      B++;
      }
  //  s_reduce[thredIdx.x] = result;
  //  __syncthreads();
   reduce_warp(&(s_reduce[warpId][0]), result);
   //   printf("here.\n");
  if (warpIndex== 0)
    {
      //      printf("writing %f to %d %d thread %d.\n", s_reduce[warpId][0], m,n , threadIdx.x) 
      fetchC(m,n) = beta * fetchC(m,n) + alpha * s_reduce[warpId][0];
      //           C[m + n * MM] = beta * C[m + n * MM] + alpha * s_reduce[warpId][0];
    }
}


__global__ void matrixMultiply_thread(int MM, int NN, int KK, float alpha, float *A, int MMM, float *B, int KKK, float beta, float *C, int NNN)
{
  //  printf("here0.\n");
  //    __shared__ float s_reduce[WARPS_PER_BLOCK][WARP_SIZE];
    //  __shared__ float s_reduce[WARP_SIZE][WARPS_PER_BLOCK];
  __shared__ float s_A[16][17];
  __shared__ float s_B[16][17];
  int m = blockIdx.x * BLOCK_SIZE_SQROOT + threadIdx.x;
  int n = blockIdx.y * BLOCK_SIZE_SQROOT + threadIdx.y;
  //  printf("m %d n %d.\n", m, n);
  if (n >= NN || m >= MM) return;

  float result = 0.0f;
  int MM_ = (MM>>4) << 4;
  int NN_ = (NN>>4) << 4;
  if ((MM_ != MM && m>=MM_) || (NN_ != NN && n>=NN_) )// if the last few 
    {
      for (int k = 0; k < KK; k ++)
        {
          result += fetchA(m,k) * fetchB(k,n);
        }
    }
  else   // use shared memory
    {
      //      printf("here.\n");
      int KK_ = (KK >> 4) << 4;
      for (int k = 0; k < KK_; k+=16)
        {
          //          printf("thread %d %d loading A %f(%d %d) B %f(%d %d).\n", threadIdx.x, threadIdx.y, fetchA(m, k+threadIdx.y), m, k+threadIdx.y, fetchB(k+threadIdx.x, n), k+threadIdx.x, n);
          s_A[threadIdx.x][threadIdx.y] = fetchA(m ,k+threadIdx.y);
          s_B[threadIdx.x][threadIdx.y] = fetchB(k+threadIdx.x, n);
          __syncthreads();
          #pragma unroll
          for (int i = 0; i != 16; i++)
            result += s_A[threadIdx.x][i]  * s_B[i][threadIdx.y];
          __syncthreads();
        }
      while (KK_ < KK)  // the remaining
        {
          result += fetchA(m, KK_) * fetchB(KK_, n);
          KK_++;
        }
      // the remaining
    }

 fetchC(m,n) = beta * fetchC(m,n) + alpha * result;

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

#if 0
#endif

