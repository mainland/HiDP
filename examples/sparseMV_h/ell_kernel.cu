#include "global.h"
#include "template/reduce.cuh"

__launch_bounds__(WARP_KERNEL_SIZE,2)
__global__ void ell_kernel_warp(int *col_ptr,  const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows, int cols)
{
  __shared__ DTYPE s_sum[WARP_KERNEL_WARP_SIZE][32];
  int numWarps = (gridDim.x * blockDim.x) >> 5;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int startId = gid >> 5;
  int warpId = threadIdx.x & 0x1F;
  for (int id = startId; id < rows; id += numWarps)
    {
      int *col_ptr1 = col_ptr + cols * id;
      const DTYPE *data_ptr1 = data + cols * id;
      DTYPE sum = AddOP<DTYPE>::identity();

      for (int myid = warpId; myid < cols; myid += 32)
        {
          int index = col_ptr1[myid];


          if (index == -1)
            sum += AddOP<DTYPE>::identity();
          else 
            {
              DTYPE xx = x[index];
              sum += xx * data_ptr1[myid];
            }
        }
      sum = reduce_warp<AddOP<DTYPE>, DTYPE>(sum, &s_sum[threadIdx.x>>5][0], warpId);
      if (warpId == 0)
        {
          y_ptr[id] = s_sum[threadIdx.x>>5][0];

        }
    }
}


__launch_bounds__(WARP_KERNEL_SIZE,2)
__global__ void ell_kernel_subwarp(int *col_ptr,  const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows, int cols)
{
  __shared__ DTYPE s_sum[WARP_KERNEL_SUBWARP_NUM][WARP_KERNEL_SUBWARP_SIZE];
  int numWarps = (gridDim.x * blockDim.x) >> 3;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int startId = gid >> 3;
  int warpId = threadIdx.x & 0x7;
  for (int id = startId; id < rows; id += numWarps)
    {
      int *col_ptr1 = col_ptr + cols * id;
      const DTYPE *data_ptr1 = data + cols * id;
      DTYPE sum = AddOP<DTYPE>::identity();

      for (int myid = warpId; myid < cols; myid += 8)
        {
          int index = col_ptr1[myid];


          if (index == -1)
            sum += AddOP<DTYPE>::identity();
          else 
            {
              DTYPE xx = x[index];
              sum += xx * data_ptr1[myid];
            }
        }
      sum = reduce_subwarp<AddOP<DTYPE>, DTYPE, 8>(sum, &s_sum[threadIdx.x>>3][0], warpId);
      if (warpId == 0)
        y_ptr[id] = s_sum[threadIdx.x>>3][0];
    }
}




__launch_bounds__(WARP_KERNEL_SIZE,2)
__global__ void ell_kernel_thread(int *col_ptr,  const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows, int cols)
{
  int numWarps = gridDim.x * blockDim.x;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int startId = gid;
  int warpId = 0;
  for (int id = startId; id < rows; id += numWarps)
    {
      int *col_ptr1 = col_ptr + cols * id;
      const DTYPE *data_ptr1 = data + cols * id;
      DTYPE sum = AddOP<DTYPE>::identity();

      for (int myid = warpId; myid < cols; myid += 1)
        {
          int index = col_ptr1[myid];
          if (index == -1)
            break;
          else 
            {
              DTYPE xx = x[index];
              sum += xx * data_ptr1[myid];
            }
        }
      y_ptr[id] = sum;
    }
  
}
