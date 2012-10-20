#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cusp/hyb_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/multiply.h>

#include "ScopeProfile.h"
#include "global.h"
#include "HiArray.h"
#include "kernel.h"


extern std::vector<StepProfile> gProfile;
extern std::map<std::string, long> gProfileGroup;

using namespace cusp;
void runTest(int argc, char **argv)
{
  gProfileGroup.clear();
  for (int mtx = 1; mtx < argc; mtx++)
    {
  gProfile.clear();


  //  cusp::csr_matrix<int, DTYPE, cusp::host_memory> A(ROWS, COLS, ENTRIES);
  cusp::csr_matrix<int, DTYPE, host_memory> A;

    // load a matrix stored in MatrixMarket format
  //  cusp::io::read_matrix_market_file(A, "cusp1/cant.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/consph.mtx");
  //
   cusp::io::read_matrix_market_file(A, argv[mtx]);
  //  cusp::io::read_matrix_market_file(A, "cusp1/qcd5_4.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/rail4284.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/rma10.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/webbase-1M.mtx");

  
  thrust::host_vector<int> begin(A.num_rows+1);
  thrust::host_vector<DTYPE> xx(A.num_cols);

  for (int i = 0; i < A.num_rows+1; i++)
      begin[i] = A.row_offsets[i];

  for (int i = 0; i < A.num_cols; i++)
      xx[i] = 1.0f;

 printf("-------------------\n");
 printf("matrix %s rows %d cols %d num entries %d avg %d.\n", argv[mtx], A.num_rows, A.num_cols, A.num_entries, A.num_entries/A.num_rows); 

 //  int totalSize = begin[A.num_cols];
  thrust::host_vector<int> col_index(A.num_entries);
  thrust::host_vector<DTYPE> data(A.num_entries);
  for (int i = 0; i < A.num_entries; i++)
    {
      col_index[i] = A.column_indices[i];
      data[i] = A.values[i];
    }

  
 cusp::csr_matrix<int,DTYPE,cusp::device_memory> B = A;
 cusp::array1d<DTYPE, cusp::host_memory> xh(A.num_cols);
 for (int i = 0; i < A.num_cols; i++)
   xh[i] = 1.0f;
 
 cusp::array1d<DTYPE, cusp::device_memory> xd = xh;
 cusp::array1d<DTYPE, cusp::device_memory> yd(A.num_rows);

 thrust::device_vector<int> col_index_d = col_index;
 thrust::device_vector<int> begin_d = begin;
 thrust::device_vector<DTYPE> data_d = data;
 thrust::device_vector<DTYPE> xx_d = xx;
 thrust::device_vector<DTYPE> yy_d(A.num_rows);

 int * col_ptr = thrust::raw_pointer_cast(col_index_d.data());
 int * begin_ptr = thrust::raw_pointer_cast(begin_d.data());
 DTYPE *data_ptr = thrust::raw_pointer_cast(data_d.data());
 DTYPE *yy_ptr = thrust::raw_pointer_cast(yy_d.data());
 DTYPE *xx_ptr = thrust::raw_pointer_cast(xx_d.data());

 {
   GPUScopeProfile cuspp("cusp_multiply_csr");
   cusp::multiply(B, xd, yd);
 }

 cusp::array1d<DTYPE, cusp::host_memory> yh = yd;

 {
   GPUScopeProfile cuspp("spmv_warp_csr");
   //   printf("calling spmv_kenel %p %p %p %p %p\n", col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr);
   spmv_kernel_warp<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
 }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "csr warp");
 }

 {
   GPUScopeProfile cuspp("spmv_subwarp_csr");
   //   printf("calling spmv_kenel %p %p %p %p %p\n", col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr);
   spmv_kernel_subwarp<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
 }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "csr subwarp");
 }


 {
   GPUScopeProfile cuspp("spmv_subwarp2_csr");
   //   printf("calling spmv_kenel %p %p %p %p %p\n", col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr);
   spmv_kernel_subwarp2<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
 }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "csr subwarp2");
 }


 {
   GPUScopeProfile cuspp("spmv_thread_csr");
   //   printf("calling spmv_kenel %p %p %p %p %p\n", col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr);
   spmv_kernel_thread<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
 }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "csr thread");
 }


 // cusp::array1d<DTYPE, cusp::host_memory> yh = yd;
 /* for (int i = 0; i != 20; i++)
    printf("yd[%d] = %f.\n", i, yh[i]);*/

 thrust::host_vector<DTYPE> yy_h = yy_d;
 /* for (int i = 0; i != 20; i++)
    printf("yyd[%d] = %f.\n", i, yy_h[i]);*/

 printf("\n");
 showProfileResult(gProfile);


    }

 showProfileResult(gProfileGroup);
 printf("*****************\n");
  /*  cusp::csr_matrix<int,DTYPE,cusp::device_memory> B = A;

*/

}
