#include "global.h"
#include "template/reduce.cuh"

__global__ void coo_gen_begin(int *begin_ptr, int *next_ptr, int *row_ptr,  int size, int entry_size)
{
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (; gid < entry_size - 1; gid += step)
    {
      int first = row_ptr[gid];
      int second = row_ptr[gid+1];
      if (row_ptr[gid] != row_ptr[gid+1])
        {
          begin_ptr[second] = gid+1;
        /*#if __CUDA_ARCH__ >= 200
          if (second == 1)
            printf("writing %d to index %d.\n", gid+1, 1);
          if (first == 0)
            printf("writing next %d to index %d.\n", second, 0);
            #endif*/
          next_ptr[first] = second;
        }
    }
  if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      begin_ptr[size] = entry_size;
      begin_ptr[row_ptr[0]] = 0;
      next_ptr[row_ptr[entry_size-1]] = size;
    }
}

//__launch_bounds__(WARP_KERNEL_SIZE,2)
__global__ void coo_kernel_warp(const int *col_ptr, const int *begin, const int *next, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows)
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
      if (low == -1) continue;

      int high = begin[next[id]];

      if (low >= high) continue;

      DTYPE sum = AddOP<DTYPE>::identity();
      /*     #if __CUDA_ARCH__ >= 200
      if (warpId == 0 && (low != 2000*id || high != 2000*(id+1)))
        printf("row %d, low %d high %d size %d next id %d.\n", id, low, high, high-low, next[id]);
        #endif*/
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



__launch_bounds__(WARP_KERNEL_SIZE,1)
__global__ void coo_kernel_subwarp(const int *col_ptr, const int *begin, const int *next, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows)
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
      if (low == -1) continue;

      int high = begin[next[id]];

      if (low >= high) continue;

      DTYPE sum = AddOP<DTYPE>::identity();
      /*#if __CUDA_ARCH__ >= 200
      if (high-low > 20 && warpId == 0)
        printf("high %d low %d row %d size %d.\n", high, low, id, high-low);
        #endif*/
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



__launch_bounds__(WARP_KERNEL_SIZE,1)
__global__ void coo_kernel_8subwarp(const int *col_ptr, const int *begin, const int *next, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows)
{
  __shared__ DTYPE s_sum[WARP_KERNEL_8SUBWARP_NUM][WARP_KERNEL_8SUBWARP_SIZE];
  //__shared__ DTYPE s_sum[WARP_KERNEL_SIZE];
  int numWarps = (gridDim.x * blockDim.x) >> 2;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int startId = gid >> 2;
  int warpId = threadIdx.x & 0x3;

  for (int id = startId; id < rows; id += numWarps)
    {
      int low = begin[id];
      if (low == -1) continue;

      int high = begin[next[id]];

      if (low >= high) continue;

      DTYPE sum = AddOP<DTYPE>::identity();
      /*#if __CUDA_ARCH__ >= 200
      if (high-low > 20 && warpId == 0)
        printf("high %d low %d row %d size %d.\n", high, low, id, high-low);
        #endif*/
      for (int myid = low + warpId; myid < high; myid += 4)
        {
            int index = col_ptr[myid];
            DTYPE xx = x[index];
            sum += xx * data[myid];
        }
      //         printf("mysum is %f.\n", sum);
      sum = reduce_subwarp<AddOP<DTYPE>, DTYPE, 4>(sum, &s_sum[threadIdx.x>>2][0], warpId);
      if (warpId == 0)
        y_ptr[id] = s_sum[threadIdx.x>>2][0];
      //        y_ptr[id] = s_sum[threadIdx.x];
      /*      if (id == 0 && warpId == 0)*/
      //      if (warpId == 0) printf("writing %f to id 0.\n", sum);
    }
}


__launch_bounds__(WARP_KERNEL_SIZE,1)
__global__ void coo_kernel_16subwarp(const int *col_ptr, const int *begin, const int *next, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows)
{
  __shared__ DTYPE s_sum[WARP_KERNEL_16SUBWARP_NUM][WARP_KERNEL_16SUBWARP_SIZE];
  //__shared__ DTYPE s_sum[WARP_KERNEL_SIZE];
  int numWarps = (gridDim.x * blockDim.x) >> 1;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int startId = gid >> 1;
  int warpId = threadIdx.x & 0x1;

  for (int id = startId; id < rows; id += numWarps)
    {
      int low = begin[id];
      if (low == -1) continue;

      int high = begin[next[id]];

      if (low >= high) continue;

      DTYPE sum = AddOP<DTYPE>::identity();
      /*#if __CUDA_ARCH__ >= 200
      if (high-low > 20 && warpId == 0)
        printf("high %d low %d row %d size %d.\n", high, low, id, high-low);
        #endif*/
      for (int myid = low + warpId; myid < high; myid += 2)
        {
            int index = col_ptr[myid];
            DTYPE xx = x[index];
            sum += xx * data[myid];
        }
      //         printf("mysum is %f.\n", sum);
      sum = reduce_subwarp<AddOP<DTYPE>, DTYPE, 2>(sum, &s_sum[threadIdx.x>>1][0], warpId);
      if (warpId == 0)
        y_ptr[id] = s_sum[threadIdx.x>>1][0];
      //        y_ptr[id] = s_sum[threadIdx.x];
      /*      if (id == 0 && warpId == 0)*/
      //      if (warpId == 0) printf("writing %f to id 0.\n", sum);
    }
}



__global__ void coo_kernel_thread(const int *col_ptr, const int *begin, const int *next, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows)
{
  int numWarps = (gridDim.x * blockDim.x);
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int startId = gid;
  //  int warpId = threadIdx.x;

  for (int id = startId; id < rows; id += numWarps)
    {
      int low = begin[id];
      if (low == -1) continue;

      int high = begin[next[id]];

      if (low >= high) continue;

      /*#if __CUDA_ARCH__ >= 200
      if (high-low > 20)
        printf("high %d low %d row %d size %d.\n", high, low, id, high-low);
        #endif*/

      DTYPE sum = AddOP<DTYPE>::identity();
      for (int myid = low; myid < high; myid++)
        {
            int index = col_ptr[myid];
            DTYPE xx = x[index];
            sum += xx * data[myid];
        }
      //         printf("mysum is %f.\n", sum);
      y_ptr[id] = sum;
      //        y_ptr[id] = s_sum[threadIdx.x];
      /*      if (id == 0 && warpId == 0)*/
      //      if (warpId == 0) printf("writing %f to id 0.\n", sum);
    }
}


