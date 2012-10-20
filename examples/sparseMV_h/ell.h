void runTest_ell(int argc, char **argv)
{
  // create an empty sparse matrix structure (HYB format)
  //  cusp::hyb_matrix<int, float, cusp::device_memory> A;
  /*  thrust::device_vector<int> begin(COLS+1);
  int *col_nums = new int[ROWS];
  int totalSize = 0;
  for (i = 0; i < COLS; i++)
    {
      int rowsize = rand()%ROWS *DENSITY;
      begin[i] = totalSize;
      col_nums[i] = rowsize;
      totalSize += rowsize;
    }
  begin[COLS] = totalSize;

  thrust::device_vector<int> sizes(totalSize);
  for (i = 0; i < COLS; i++)

    {
      int rowsize = col_nums[i];
      
    }
  */


  //  thrust::device_vector<int> size(bodySize);
  assert(argc == 2);

  //  cusp::csr_matrix<int, float, cusp::host_memory> A(ROWS, COLS, ENTRIES);
  cusp::ell_matrix<int, float, host_memory> A;

    // load a matrix stored in MatrixMarket format
  //  cusp::io::read_matrix_market_file(A, "cusp1/cant.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/consph.mtx");
  //
   cusp::io::read_matrix_market_file(A, argv[1]);
  //  cusp::io::read_matrix_market_file(A, "cusp1/qcd5_4.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/rail4284.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/rma10.mtx");
  //  cusp::io::read_matrix_market_file(A, "cusp1/webbase-1M.mtx");

  
  thrust::host_vector<int> begin(A.num_rows+1);
  thrust::host_vector<float> xx(A.num_cols);

  for (int i = 0; i < A.num_rows+1; i++)
      begin[i] = A.row_offsets[i];

  for (int i = 0; i < A.num_cols; i++)
      xx[i] = 1.0f;

  int totalSize = begin[A.num_cols];
  printf("rows %d cols %d num entries %d.\n", A.num_rows, A.num_cols, totalSize);
  thrust::host_vector<int> col_index(totalSize);
  thrust::host_vector<float> data(totalSize);
  for (int i = 0; i < totalSize; i++)
    {
      col_index[i] = A.column_indices[i];
      data[i] = A.values[i];
    }

  
 cusp::csr_matrix<int,float,cusp::device_memory> B = A;
 cusp::array1d<float, cusp::host_memory> xh(A.num_cols);
 for (int i = 0; i < A.num_cols; i++)
   xh[i] = 1.0f;
 
 cusp::array1d<float, cusp::device_memory> xd = xh;
 cusp::array1d<float, cusp::device_memory> yd(A.num_rows);

 thrust::device_vector<int> col_index_d = col_index;
 thrust::device_vector<int> begin_d = begin;
 thrust::device_vector<float> data_d = data;
 thrust::device_vector<float> xx_d = xx;
 thrust::device_vector<float> yy_d(A.num_rows);

 int * col_ptr = thrust::raw_pointer_cast(col_index_d.data());
 int * begin_ptr = thrust::raw_pointer_cast(begin_d.data());
 float *data_ptr = thrust::raw_pointer_cast(data_d.data());
 float *yy_ptr = thrust::raw_pointer_cast(yy_d.data());
 float *xx_ptr = thrust::raw_pointer_cast(xx_d.data());

 {
   GPUScopeProfile cuspp("spmv_warp");
   //   printf("calling spmv_kenel %p %p %p %p %p\n", col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr);
   spmv_kernel_warp<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
 }

 {
   GPUScopeProfile cuspp("spmv_subwarp");
   //   printf("calling spmv_kenel %p %p %p %p %p\n", col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr);
   spmv_kernel_subwarp<<<128, WARP_KERNEL_SIZE>>>(col_ptr, begin_ptr, data_ptr, xx_ptr, yy_ptr, A.num_rows);
 }

 {
   GPUScopeProfile cuspp("cusp_multiply");
   cusp::multiply(B, xd, yd);
 }

 cusp::array1d<float, cusp::host_memory> yh = yd;
 for (int i = 0; i != 20; i++)
   printf("yd[%d] = %f.\n", i, yh[i]);

 thrust::host_vector<float> yy_h = yy_d;
 for (int i = 0; i != 20; i++)
   printf("yyd[%d] = %f.\n", i, yy_h[i]);

 printf("\n");

 showProfileResult(gProfile);

  /*  cusp::csr_matrix<int,float,cusp::device_memory> B = A;

*/

}
