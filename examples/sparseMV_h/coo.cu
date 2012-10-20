#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>

#include <cusp/hyb_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/multiply.h>
#include "kernel.h"
#include "ScopeProfile.h"

extern std::vector<StepProfile> gProfile;
extern std::map<std::string, long> gProfileGroup;
using namespace std;
using namespace cusp;


void runTest_coo(int argc, char **argv)
{
  gProfileGroup.clear();
  for (int mtx = 1; mtx < argc; mtx++)
    {
  gProfile.clear();


#if 1
  //  cusp::csr_matrix<int, DTYPE, cusp::host_memory> A(ROWS, COLS, ENTRIES);
  cusp::coo_matrix<int, DTYPE, host_memory> A;

    // load a matrix stored in MatrixMarket format
  //  cusp::io::read_matrix_market_file(A, "cusp1/cant.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/consph.mtx");
  //
  try{
  cusp::io::read_matrix_market_file(A, argv[mtx]);
  } catch (...)
      {
        continue;
      }
#else
cusp::coo_matrix<int,float,cusp::host_memory> A(4,3,6);

// initialize matrix entries on host
A.row_indices[0] = 0; A.column_indices[0] = 0; A.values[0] = 10;
A.row_indices[1] = 0; A.column_indices[1] = 2; A.values[1] = 20;
A.row_indices[2] = 2; A.column_indices[2] = 2; A.values[2] = 30;
A.row_indices[3] = 3; A.column_indices[3] = 0; A.values[3] = 40;
A.row_indices[4] = 3; A.column_indices[4] = 1; A.values[4] = 50;
A.row_indices[5] = 3; A.column_indices[5] = 2; A.values[5] = 60;

#endif
  //  cusp::io::read_matrix_market_file(A, "cusp1/qcd5_4.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/rail4284.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/rma10.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/webbase-1M.mtx");
 printf("-------------------\n");
 printf("matrix %s rows %d cols %d num entries %d avg %d.\n", argv[mtx], A.num_rows, A.num_cols, A.num_entries, A.num_entries/A.num_rows); 
  /*printf("index .\n");
    for (int i = 0; i < 100; i++)
    {
      printf("row %d column %d.\n", A.row_indices[A.num_entries-1-i], A.column_indices[A.num_entries-1-i]);
      }*/


  thrust::host_vector<int> row_indexes(A.num_entries);
  thrust::host_vector<int> col_indexes(A.num_entries);
  thrust::host_vector<DTYPE> data(A.num_entries);
  int * row_ptr = thrust::raw_pointer_cast(row_indexes.data());
  int * col_ptr = thrust::raw_pointer_cast(col_indexes.data());
  DTYPE *data_ptr = thrust::raw_pointer_cast(data.data());
  memcpy(row_ptr, &(A.row_indices[0]), sizeof(int) * A.num_entries);
  memcpy(col_ptr, &(A.column_indices[0]), sizeof(int) * A.num_entries);
  memcpy(data_ptr, &(A.values[0]), sizeof(DTYPE) * A.num_entries);

  thrust::device_vector<int> row_index_d = row_indexes;
  thrust::device_vector<int> col_index_d = col_indexes;
  thrust::device_vector<DTYPE> data_d = data;

  // reset pointer
  row_ptr = thrust::raw_pointer_cast(row_index_d.data());
  col_ptr = thrust::raw_pointer_cast(col_index_d.data());
  data_ptr = thrust::raw_pointer_cast(data_d.data());

  cusp::array1d<DTYPE, cusp::host_memory> xh(A.num_cols);
  thrust::host_vector<DTYPE> xx(A.num_cols);
  for (int i = 0; i < A.num_cols; i++)
    {
      xh[i] = i+1;
      xx[i] = i+1;
    }

  cusp::array1d<DTYPE, cusp::device_memory> xd = xh;
  cusp::array1d<DTYPE, cusp::device_memory> yd(A.num_rows);
  cusp::coo_matrix<int,DTYPE, device_memory> B = A;  
  {
    GPUScopeProfile cuspp("coo_cusp_multiply");
    cusp::multiply(B, xd, yd);
  }


 cusp::array1d<DTYPE, cusp::host_memory> yh = yd;

  thrust::device_vector<DTYPE> xx_d = xx;
  thrust::device_vector<DTYPE> yy_d(A.num_rows);

  
  thrust::device_vector<int> begin(A.num_rows+1);
  thrust::device_vector<int> next(A.num_rows);
  int *begin_ptr = thrust::raw_pointer_cast(begin.data());
  int *next_ptr = thrust::raw_pointer_cast(next.data());

 // thrust::generate(yy_d.begin(), yy_d.end(), rand);

 DTYPE *yy_ptr = thrust::raw_pointer_cast(yy_d.data());
 DTYPE *xx_ptr = thrust::raw_pointer_cast(xx_d.data());

  {
    GPUScopeProfile cuspp("coo_hidp_multiply_thread");
    cudaMemset(begin_ptr, 0xFF, sizeof(int) * (A.num_rows+1));        
    cudaMemset(next_ptr, 0, sizeof(int) * (A.num_rows));    
    coo_gen_begin<<<128, WARP_KERNEL_SIZE>>>(begin_ptr, next_ptr, row_ptr, A.num_rows,A.num_entries);
     coo_kernel_thread<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, next_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
  }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "coo thread");
 }


  {
    GPUScopeProfile cuspp("coo_hidp_multiply_warp");
    cudaMemset(begin_ptr, 0xFF, sizeof(int) * (A.num_rows+1));        
    cudaMemset(next_ptr, 0, sizeof(int) * (A.num_rows));    
    coo_gen_begin<<<128, WARP_KERNEL_SIZE>>>(begin_ptr, next_ptr, row_ptr, A.num_rows,A.num_entries);
    coo_kernel_warp<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, next_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
  }

  /*thrust::host_vector<int> begin_h = begin;
  thrust::host_vector<int> next_h = next;
  printf("begin:\n");
  for (int i = 0; i != A.num_rows+1; i++)
    printf("%d ", begin_h[i]);
  printf("\n");
  printf("next:\n");
  for (int i = 0; i != A.num_rows; i++)
    printf("%d ", next_h[i]);
    printf("\n");*/
 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "coo warp");
 }

 // thrust::generate(yy_d.begin(), yy_d.end(), rand);
  {
    GPUScopeProfile cuspp("coo_hidp_multiply_subwarp");
    cudaMemset(begin_ptr, 0xFF, sizeof(int) * (A.num_rows+1));        
    cudaMemset(next_ptr, 0, sizeof(int) * (A.num_rows));    
    coo_gen_begin<<<128, WARP_KERNEL_SIZE>>>(begin_ptr, next_ptr, row_ptr, A.num_rows,A.num_entries);
    coo_kernel_subwarp<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, next_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
  }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "coo sub warp");
 }

 // thrust::generate(yy_d.begin(), yy_d.end(), rand);
  {
    GPUScopeProfile cuspp("coo_hidp_multiply_8subwarp");
    cudaMemset(begin_ptr, 0xFF, sizeof(int) * (A.num_rows+1));        
    cudaMemset(next_ptr, 0, sizeof(int) * (A.num_rows));    
    coo_gen_begin<<<128, WARP_KERNEL_SIZE>>>(begin_ptr, next_ptr, row_ptr, A.num_rows,A.num_entries);
    coo_kernel_8subwarp<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, next_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
  }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "coo 8 sub warp");
 }


 // thrust::generate(yy_d.begin(), yy_d.end(), rand);
  {
    GPUScopeProfile cuspp("coo_hidp_multiply_16subwarp");
    cudaMemset(begin_ptr, 0xFF, sizeof(int) * (A.num_rows+1));        
    cudaMemset(next_ptr, 0, sizeof(int) * (A.num_rows));    
    coo_gen_begin<<<128, WARP_KERNEL_SIZE>>>(begin_ptr, next_ptr, row_ptr, A.num_rows,A.num_entries);
    coo_kernel_16subwarp<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, next_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
  }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "coo 16 sub warp");
 }

 // thrust::generate(yy_d.begin(), yy_d.end(), rand);

 /* for (int i = 0; i != 20; i++)
    printf("yyd[%d] = %f.\n", i, yy_h[i]);*/
 showProfileResult(gProfile);
    }
  
  printf("****** in summary *******:\n");

  showProfileResult(gProfileGroup);
 printf("*****************\n");
  /*  cusp::csr_matrix<int,DTYPE,cusp::device_memory> B = A;

*/

}
