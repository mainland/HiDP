#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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


void runTest_ell(int argc, char **argv)
{

  gProfileGroup.clear();

  for (int mtx = 1; mtx < argc; mtx++)
    {
  gProfile.clear();

  //  cusp::csr_matrix<int, DTYPE, cusp::host_memory> A(ROWS, COLS, ENTRIES);
  cusp::csr_matrix<int, DTYPE, host_memory> AA;
  cusp::ell_matrix<int, DTYPE, host_memory> A;

    // load a matrix stored in MatrixMarket format
  //  cusp::io::read_matrix_market_file(A, "cusp1/cant.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/consph.mtx");
  //
  //  try{
  cusp::io::read_matrix_market_file(AA, argv[mtx]);
  A = AA;
  //  } catch (...)
  //      {
  //        continue;
  //      }
  //  cusp::io::read_matrix_market_file(A, "cusp1/qcd5_4.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/rail4284.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/rma10.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/webbase-1M.mtx");
  
  thrust::host_vector<int> indexes(A.column_indices.num_rows * A.column_indices.num_cols);
  thrust::host_vector<DTYPE> data(A.column_indices.num_rows * A.column_indices.num_cols);
 printf("-------------------\n");
 printf("matrix %s rows %d cols %d num entries %d avg %d.\n", argv[mtx], A.num_rows, A.num_cols, A.num_entries, A.num_entries/A.num_rows); 

  int * col_ptr = thrust::raw_pointer_cast(indexes.data());
  DTYPE *data_ptr = thrust::raw_pointer_cast(data.data());
  memcpy(col_ptr, &(A.column_indices(0,0)), sizeof(int) * A.column_indices.num_cols * A.column_indices.num_rows);
  for (int j = 0; j != A.column_indices.num_rows; j++)
    for (int i = 0; i != A.column_indices.num_cols; i++)
    {
      //      printf("%d ", A.column_indices(0, i));
      /*      if (A.column_indices(j,i) == -1)
              printf("-1 found in %d %d.\n", j,i);*/
      *col_ptr = A.column_indices(j,i);
      *data_ptr = A.values(j,i);
      col_ptr++; data_ptr++;
    }

  thrust::device_vector<int> col_index_d = indexes;
  thrust::device_vector<DTYPE> data_d = data;

  // reset pointer
  col_ptr = thrust::raw_pointer_cast(col_index_d.data());
  data_ptr = thrust::raw_pointer_cast(data_d.data());
  //  printf("here 2.\n");
  //  for (int i = 0; i  < 100; i++)
  //      cout << indexes[i] << " ";

  cusp::array1d<DTYPE, cusp::host_memory> xh(A.num_cols);
  thrust::host_vector<DTYPE> xx(A.num_cols);
  for (int i = 0; i < A.num_cols; i++)
    {
      xh[i] = 1.0; //i+1;
      xx[i] = 1.0; // i+1;
    }

  cusp::array1d<DTYPE, cusp::device_memory> xd = xh;
  cusp::array1d<DTYPE, cusp::device_memory> yd(A.num_rows);
  cusp::ell_matrix<int,DTYPE, device_memory> B = A;  
  {
    GPUScopeProfile cuspp("ell_cusp_multiply");
    cusp::multiply(B, xd, yd);
  }

 cusp::array1d<DTYPE, cusp::host_memory> yh = yd;
 /*  for (int i = 0; i != 20; i++)
     printf("yd[%d] = %f.\n", i, yh[i]);*/

  thrust::device_vector<DTYPE> xx_d = xx;
  thrust::device_vector<DTYPE> yy_d(A.num_rows);

 DTYPE *yy_ptr = thrust::raw_pointer_cast(yy_d.data());
 DTYPE *xx_ptr = thrust::raw_pointer_cast(xx_d.data());
  {
    GPUScopeProfile cuspp("ell_hidp_multiply_warp");
    ell_kernel_warp<<<128, WARP_KERNEL_SIZE>>>(col_ptr, data_ptr, xx_ptr, yy_ptr, A.column_indices.num_rows, A.column_indices.num_cols);
  
  }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "csr warp");
 }

  {
    GPUScopeProfile cuspp("ell_hidp_multiply_subwarp");
    ell_kernel_subwarp<<<128, WARP_KERNEL_SIZE>>>(col_ptr, data_ptr, xx_ptr, yy_ptr, A.column_indices.num_rows, A.column_indices.num_cols);
  
  }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "csr warp");
 }

 {
    GPUScopeProfile cuspp("ell_hidp_multiply_thread");
    ell_kernel_thread<<<128, WARP_KERNEL_SIZE>>>(col_ptr, data_ptr, xx_ptr, yy_ptr, A.column_indices.num_rows, A.column_indices.num_cols);
  }

 {
   thrust::host_vector<DTYPE> yy_h = yy_d;
   isIdentical(&yh[0], &yy_h[0], A.num_rows, "csr warp");
 }


 showProfileResult(gProfile);

  /*  cusp::csr_matrix<int,DTYPE ,cusp::device_memory> B = A;

*/

    }
 showProfileResult(gProfileGroup);
 printf("*****************\n");
}
