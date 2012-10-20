#include "global.h"
#include "template/reduce.cuh"

__global__ void spmv_kernel_warp(int *col_ptr, int *begin, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows)
{
  __shared__ DTYPE s_sum[WARP_KERNEL_WARP_SIZE][32];
  //__shared__ DTYPE s_sum[WARP_KERNEL_SIZE];
  int numWarps = (gridDim.x * blockDim.x) >> 5;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int startId = gid >> 5;
  int warpId = threadIdx.x & 0x1F;
  for (int id = startId; id < rows; id += numWarps)
    {
      int low = begin[id];
      int high = begin[id+1];

      //      if (low >= high) continue;

      DTYPE sum = AddOP<DTYPE>::identity();
      for (int myid = low + warpId; myid < high; myid += 32)
        {
            int index = col_ptr[myid];
            DTYPE xx = x[index];
            sum += xx * data[myid];
        }
      //      printf("mysum is %f.\n", sum);
      sum = reduce_warp<AddOP<DTYPE>, DTYPE>(sum, &s_sum[threadIdx.x>>5][0], warpId);
      if (warpId == 0)
        y_ptr[id] = s_sum[threadIdx.x>>5][0];
      //        y_ptr[id] = s_sum[threadIdx.x];
      /*      if (id == 0 && warpId == 0)
              printf("writing %f to id 0.\n", sum);*/
    }
}



__global__ void spmv_kernel_subwarp(int *col_ptr, int *begin, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows)
{
  __shared__ DTYPE s_sum[WARP_KERNEL_SUBWARP_NUM][WARP_KERNEL_SUBWARP_SIZE];
  //__shared__ DTYPE s_sum[WARP_KERNEL_SIZE];
  int numWarps = (gridDim.x * blockDim.x) >> 3;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int startId = gid >> 3;
  int warpId = threadIdx.x & 0x7;

  for (int id = startId; id < rows; id += numWarps)
    {
      int low = begin[id];
      int high = begin[id+1];

      //      if (low >= high) continue;

      DTYPE sum = AddOP<DTYPE>::identity();
      for (int myid = low + warpId; myid < high; myid += 8)
        {
            int index = col_ptr[myid];
            DTYPE xx = x[index];
            sum += xx * data[myid];
        }
      //         printf("mysum is %f.\n", sum);
      sum = reduce_subwarp<AddOP<DTYPE>, DTYPE, 8>(sum, &s_sum[threadIdx.x>>3][0], warpId);
      if (warpId == 0)
        y_ptr[id] = s_sum[threadIdx.x>>3][0];
      //        y_ptr[id] = s_sum[threadIdx.x];
      /*      if (id == 0 && warpId == 0)*/
      //      if (warpId == 0) printf("writing %f to id 0.\n", sum);
    }
}



__global__ void spmv_kernel_subwarp2(int *col_ptr, int *begin, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows)
{
  __shared__ DTYPE s_sum[WARP_KERNEL_SUBWARP2_NUM][WARP_KERNEL_SUBWARP2_SIZE];
  //__shared__ DTYPE s_sum[WARP_KERNEL_SIZE];
  int numWarps = (gridDim.x * blockDim.x) >> 2;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int startId = gid >> 2;
  int warpId = threadIdx.x & 0x3;

  for (int id = startId; id < rows; id += numWarps)
    {
      int low = begin[id];
      int high = begin[id+1];

      //      if (low >= high) continue;

      DTYPE sum = AddOP<DTYPE>::identity();
      for (int myid = low + warpId; myid < high; myid += 4)
        {
            int index = col_ptr[myid];
            DTYPE xx = x[index];
            sum += xx * data[myid];
        }
      //         printf("mysum is %f.\n", sum);
      sum = reduce_subwarp<AddOP<DTYPE>, DTYPE, 8>(sum, &s_sum[threadIdx.x>>3][0], warpId);
      if (warpId == 0)
        y_ptr[id] = s_sum[threadIdx.x>>2][0];
      //        y_ptr[id] = s_sum[threadIdx.x];
      /*      if (id == 0 && warpId == 0)*/
      //      if (warpId == 0) printf("writing %f to id 0.\n", sum);
    }
}



__global__ void spmv_kernel_thread(int *col_ptr, int *begin, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows)
{
  //  __shared__ DTYPE s_sum[WARP_KERNEL_WARP_SIZE][32];
  //__shared__ DTYPE s_sum[WARP_KERNEL_SIZE];
  int numWarps = (gridDim.x * blockDim.x);
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int startId = gid ;
  for (int id = startId; id < rows; id += numWarps)
    {
      int low = begin[id];
      int high = begin[id+1];

      //if (low >= high) continue;

      DTYPE sum = AddOP<DTYPE>::identity();
      for (int myid = low; myid < high; myid ++)
        {
            int index = col_ptr[myid];
            DTYPE xx = x[index];
            sum += xx * data[myid];
        }
      y_ptr[id] = sum;
    }
}


