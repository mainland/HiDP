#pragma once
#include "global.h"

__global__ void spmv_kernel_warp(int *col_ptr, int *begin, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows);

__global__ void spmv_kernel_subwarp(int *col_ptr, int *begin, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows);
__global__ void spmv_kernel_subwarp2(int *col_ptr, int *begin, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows);
__global__ void spmv_kernel_thread(int *col_ptr, int *begin, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows);


__global__ void ell_kernel_warp(int *col_ptr,  const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows, int cols);
__global__ void ell_kernel_subwarp(int *col_ptr,  const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows, int cols);
__global__ void ell_kernel_thread(int *col_ptr,  const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows, int cols);


__global__ void coo_gen_begin(int *begin_ptr, int *next_ptr, int *row_ptr,  int size, int entry_size);
__global__ void coo_kernel_warp(const int *col_ptr, const int *begin, const int *next, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows);
__global__ void coo_kernel_subwarp(const int *col_ptr, const int *begin, const int *next, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows);
__global__ void coo_kernel_8subwarp(const int *col_ptr, const int *begin, const int *next, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows);
__global__ void coo_kernel_16subwarp(const int *col_ptr, const int *begin, const int *next, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows);
__global__ void coo_kernel_thread(const int *col_ptr, const int *begin, const int *next, const DTYPE *data, const DTYPE *x, DTYPE *y_ptr, int rows);

