#include "matrixMul_kernel.cu"
#include "ScopeProfile.h"
extern std::vector<StepProfile> gProfile;
extern std::map<std::string, long> gProfileGroup;


void runTest(int argc, char **argv, int M, int K, int N)
{
#if 1
  printf("************\n M = %d N = %d K = %d.\n", M, N, K);
  gProfile.clear();
   cublasStatus status;
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
        return;
    }

    /*    thrust::host_vector<float> A(M*K);
          thrust::host_vector<float> B(K*N);*/

    thrust::device_vector<float> A(M*K);
    thrust::device_vector<float> B(K*N);
#if 1
    thrust::sequence(A.begin(), A.end());
    thrust::sequence(B.begin(), B.end());
#else
    for (int i = 0; i < M*K; i++)
      A[i] = rand()&1;
    for (int i = 0; i < N*K; i++)
      B[i] = rand()&1;
#endif

    thrust::device_vector<float> C(M*N);

    thrust::device_vector<float> C_block = C;
    thrust::device_vector<float> C_warp = C;
    thrust::device_vector<float> C_subwarp = C;
    thrust::device_vector<float> C_thread = C;    

    /* Clear last error */
    cublasGetError();
    float alpha = 1.0f;
    float beta = 0.0f;
    float *d_A = thrust::raw_pointer_cast(&A[0]);
    float *d_B = thrust::raw_pointer_cast(&B[0]);
    float *d_C = thrust::raw_pointer_cast(&C[0]);
    float *d_C_block = thrust::raw_pointer_cast(&C_block[0]);
    float *d_C_warp = thrust::raw_pointer_cast(&C_warp[0]);
    float *d_C_thread = thrust::raw_pointer_cast(&C_thread[0]);

    long flops = M*N*K ;
    /* Performs operation using cublas */
    {
      GPUScopeProfile cublas("cublas");
      cublasSgemm('n', 'n', M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
    }

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! kernel execution error.\n");
        return;
    }

    //    printf("matrix multiply cublas M %d N %d K %d gflops %f.\n", M, N, K, (M*N*K) / seconds * 10e-9);

#if 0
    {
      GPUScopeProfile mm_block("block");
    //    int warps_per_block = 8;
    dim3 blocks(M, N, 1);
    matrixMultiply_block<<<blocks, 256>>>(M, N, K, alpha, d_A, M, d_B, K, beta, d_C_block, N);
    
    //    float seconds = elapsed.seconds_elapsed();
    //    printf("matrix multiply block per element M %d N %d K %d gflops %f.\n", M, N, K, (M*N*K) / seconds * 10e-9);
    
    }


    {
      GPUScopeProfile mm_warp("warp");
    //    int warps_per_block = 8;
      dim3 blocks(M, max((N-1)/WARPS_PER_BLOCK+1,1), 1);
      matrixMultiply_warp<<<blocks, BLOCK_SIZE>>>(M, N, K, alpha, d_A, M, d_B, K, beta, d_C_warp, N);
    
    //    float seconds = elapsed.seconds_elapsed();
    //    printf("matrix multiply warp per element M %d N %d K %d gflops %f.\n", M, N, K, (M*N*K) / seconds * 10e-9);
    
    }
#endif

    {
      GPUScopeProfile mm_thread("thread");
    //    int warps_per_block = 8;
      dim3 blocks((M-1)/BLOCK_SIZE_SQROOT+1,  (N-1)/BLOCK_SIZE_SQROOT+1, 1);
      dim3 threads(16, 16, 1);
      matrixMultiply_thread<<<blocks, threads>>>(M, N, K, alpha, d_A, M, d_B, K, beta, d_C_thread, N);
    
    //    float seconds = elapsed.seconds_elapsed();
    //    printf("matrix multiply warp per element M %d N %d K %d gflops %f.\n", M, N, K, (M*N*K) / seconds * 10e-9);
    
    }

    thrust::host_vector<float> h_C = C;
    thrust::host_vector<float> h_A = A;
    thrust::host_vector<float> h_B = B;
    thrust::host_vector<float> h_C_block = C_block;
    thrust::host_vector<float> h_C_warp = C_warp;
    thrust::host_vector<float> h_C_thread = C_thread;
    /*
    printf("%f %f %f %f %f %f.\n", h_A[0], h_A[1], h_A[2], h_A[3], h_A[4], h_A[5]);
    printf("%f %f %f %f %f %f.\n", h_B[0], h_B[1], h_B[2], h_B[3], h_B[4], h_B[5]);
    printf("%f %f %f %f %f %f.\n", h_C[0], h_C[1], h_C[2], h_C[3], h_C[4], h_C[5]);
    printf("%f %f %f %f %f %f.\n", h_C_block[0], h_C_block[1], h_C_block[2], h_C_block[3], h_C_block[4], h_C_block[5]);
    printf("%f %f %f %f %f %f.\n", h_C_warp[0], h_C_warp[1], h_C_warp[2], h_C_warp[3], h_C_warp[4], h_C_warp[5]);*/

    // checking
#if 0
    {
    thrust::device_vector<float> residue(1);
    float *d_residue = thrust::raw_pointer_cast(&residue[0]);
    check_result<<<1, 512>>>(d_residue, d_C, d_C_block, N*M);
    thrust::host_vector<float> h_residue = residue;
    if (h_residue[0] > 0.01)
      printf("block diff !!!!!!!!  %f.\n", h_residue[0]);
    }
    {
    thrust::device_vector<float> residue(1);
    float *d_residue = thrust::raw_pointer_cast(&residue[0]);
    check_result<<<1, 512>>>(d_residue, d_C, d_C_warp, N*M);
    thrust::host_vector<float> h_residue = residue;
    if (h_residue[0] > 0.01)
      printf("warp diff !!!!!!!! %f.\n", h_residue[0]);
    }


    {
    thrust::device_vector<float> residue(1);
    float *d_residue = thrust::raw_pointer_cast(&residue[0]);
    check_result<<<1, 512>>>(d_residue, d_C, d_C_thread, N*M);
    thrust::host_vector<float> h_residue = residue;
    if (h_residue[0] > 0.01)
      printf("thread diff !!!!!!! %f.\n", h_residue[0]);
    }
#endif
 showProfileResult(gProfile);
}

#endif
