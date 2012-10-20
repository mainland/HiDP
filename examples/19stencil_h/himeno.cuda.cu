#include <stdio.h>
#include "himeno.h"

#define WARP_SIZE 32
/************************ segmented reduce *****************************/
template <typename T, class OP>
__device__ static inline void reduceWarp(int idx, volatile T* sdata)
{
  if ((idx & (WARP_SIZE-1)) < 16)
    {
      sdata[idx] = OP::apply(sdata[idx], sdata[idx + 16]);
      sdata[idx] = OP::apply(sdata[idx], sdata[idx +  8]);
      sdata[idx] = OP::apply(sdata[idx], sdata[idx +  4]);
      sdata[idx] = OP::apply(sdata[idx], sdata[idx +  2]);
      sdata[idx] = OP::apply(sdata[idx], sdata[idx +  1]);
    }
}



template <typename T>
__device__ inline void reduce_warp(int idx, T* sdata)
{
  if (idx < 16)
    {
      sdata[idx] += sdata[idx + 16];
      sdata[idx] += sdata[idx + 8];
      sdata[idx] += sdata[idx + 4];
      sdata[idx] += sdata[idx + 2];
      sdata[idx] += sdata[idx + 1];
    }
}

/* block-level reduction */
template <typename T, int BLOCK_SIZE>
__device__ inline void reduce_block(int idx, T *sdata)
{
  __syncthreads();
  int stride = (BLOCK_SIZE>>1);
  while (stride < WARP_SIZE)
    {
      if (idx < stride)
        sdata[idx] += sdata[idx + stride];
      stride >>= 1;
      __syncthreads();
    }
    reduce_warp<T>(idx, sdata);
}    


// Allow for swapping x and y dimensions of the CTA grid
#undef SWAP_XY

// Allow for fetching global memory via the texture cache
#define USE_TEXTURE
#undef USE_TEXTURE

// Don't allow FMAD/FFMA merging during residual computation
#undef NEW_GOSA

#if defined(USE_TEXTURE)
texture<float, 1, cudaReadModeElementType> ptex;
#define fetch_p(i) tex1Dfetch(ptex,(i))
#else
#define fetch_p(i) p[i]
#endif

__global__ void sum_gosa (float* gosa, int n)
{
    int index = threadIdx.x;
    int tx = threadIdx.x;
    __shared__ float gosa_sh[512];
    gosa_sh[tx] = 0.0f;
    __syncthreads();
    while (index < n) {
        gosa_sh[tx] += gosa[index];
        index += 512;
    }
    __syncthreads();
    if (tx < 256) gosa_sh[tx] += gosa_sh[tx+256]; __syncthreads();
    if (tx < 128) gosa_sh[tx] += gosa_sh[tx+128]; __syncthreads();
    if (tx <  64) gosa_sh[tx] += gosa_sh[tx+ 64]; __syncthreads();
    if (tx <  32) { //single warp, no if's needed
        gosa_sh[tx] += gosa_sh[tx+32]; __syncthreads();
        gosa_sh[tx] += gosa_sh[tx+16]; __syncthreads();
        gosa_sh[tx] += gosa_sh[tx+ 8]; __syncthreads();
        gosa_sh[tx] += gosa_sh[tx+ 4]; __syncthreads();
        gosa_sh[tx] += gosa_sh[tx+ 2]; __syncthreads();
        gosa_sh[tx] += gosa_sh[tx +1]; __syncthreads();
    }
    if (tx == 0) gosa[0] = gosa_sh[0];
}

__global__ void
jacobi_kernel_btm_even (const float* RESTRICT a0d, const float* RESTRICT a1d, 
                        const float* RESTRICT a2d, const float* RESTRICT a3d,
                        const float* RESTRICT b0d, const float* RESTRICT b1d,
                        const float* RESTRICT b2d, const float* RESTRICT c0d,
                        const float* RESTRICT c1d, const float* RESTRICT c2d,
                        const float* RESTRICT wrk1,const float* RESTRICT bnd,
                        const float* RESTRICT p, float* RESTRICT wrk2,
                        float* RESTRICT gosa_o, float omega)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
#if !defined(SWAP_XY)
  int gx = tx + blockIdx.x*BLOCK_X;
  int gy = ty + blockIdx.y*BLOCK_Y+1;
#else
  int gx = tx + blockIdx.y*BLOCK_X;
  int gy = ty + blockIdx.x*BLOCK_Y+1;
#endif
  int index = gx + rsize*gy;
  int btm=0,mid=1,top;
  float a0,a1,a2,a3,b0,b1,b2,c0,c1,c2,s0,ss,work1,bound,gosa=0.0f;
  __shared__ float p_sh[3][(BLOCK_Y+2)][(BLOCK_X+2)];
#if (BLOCK_Y < 2)
#error code requires BLOCK_Y >= 2
#endif
  /* Avoid accessing memory outside the array */
  if ((index-rsize-1) >= 0) p_sh[btm][ty  ][tx  ] = fetch_p(index-rsize-1);
  p_sh[btm][ty  ][tx+2] = fetch_p(index-rsize+1);
  p_sh[btm][ty+2][tx  ] = fetch_p(index+rsize-1);
  p_sh[btm][ty+2][tx+2] = fetch_p(index+rsize+1);
  p_sh[mid][ty  ][tx  ] = fetch_p(index-rsize-1+psize);
  p_sh[mid][ty  ][tx+2] = fetch_p(index-rsize+1+psize);
  p_sh[mid][ty+2][tx  ] = fetch_p(index+rsize-1+psize);
  p_sh[mid][ty+2][tx+2] = fetch_p(index+rsize+1+psize);

  for(int z=1; z<MIDPLANE_EVEN; z++)
  {
    top = (z+1)%3;
    mid = (z  )%3;
    btm = (z-1)%3;
    index += psize;
    __syncthreads();
    //load next plane
    p_sh[top][ty  ][tx  ] = fetch_p(index+psize-rsize-1);
    p_sh[top][ty  ][tx+2] = fetch_p(index+psize-rsize+1);
    p_sh[top][ty+2][tx  ] = fetch_p(index+psize+rsize-1);
    p_sh[top][ty+2][tx+2] = fetch_p(index+psize+rsize+1);
    __syncthreads();

    //apply stencil
    a0 = a0d[index];
    a1 = a1d[index];

    work1 = wrk1[index];
    s0 = work1 + a0*p_sh[top][ty+1][tx+1] + a1*p_sh[mid][ty+2][tx+1];
    b0 = b0d[index];
    a2 = a2d[index];
    s0 += b0*( p_sh[top][ty+2][tx+1] - p_sh[top][ty][tx+1] - p_sh[btm][ty+2][tx+1] + p_sh[btm][ty][tx+1] ) + a2*p_sh[mid][ty+1][tx+2];
    b1 = b1d[index];
    s0 += b1*( p_sh[mid][ty+2][tx+2] - p_sh[mid][ty][tx+2] - p_sh[mid][ty+2][tx] + p_sh[mid][ty][tx] );
    b2 = b2d[index];
    c0 = c0d[index];
    s0 += b2*( p_sh[top][ty+1][tx+2] - p_sh[btm][ty+1][tx+2] - p_sh[top][ty+1][tx] + p_sh[btm][ty+1][tx] ) + c0*p_sh[btm][ty+1][tx+1];
    c1 = c1d[index];
    c2 = c2d[index];

    s0 += c1*p_sh[mid][ty][tx+1] + c2*p_sh[mid][ty+1][tx];
    a3 = a3d[index];
    bound = bnd[index];
#if defined(NEW_GOSA)
    ss = __fmul_rn(bound,__fadd_rn(__fmul_rn(s0,a3),-p_sh[mid][ty+1][tx+1]));
#else
    ss = bound*(s0*a3 - p_sh[mid][ty+1][tx+1]);
#endif

    //write result
    if(gx<MKMAX-3&&gy<MJMAX-2)
    {
      wrk2[index] = p_sh[mid][ty+1][tx+1] + omega*ss;
      gosa += ss*ss;
    }
  }//end loop over z
  // Wait with p_sh update until all threads have finished reading p_sh
  __syncthreads();
  p_sh[0][ty][tx] = gosa;

  // Once all threads have written gosa to p_sh, reduce gosa in shared memory
  __syncthreads();
  /*n
#if (BLOCK_X > 128) || (BLOCK_X < 32) || ((BLOCK_X & (BLOCK_X - 1)) != 0)
#error code requires that BLOCK_X is a power of 2, and 32 <= BLOCK_X <= 128
#endif
#if (BLOCK_Y > 4) 
#error code requires that BLOCK_Y is <= 4
#endif
  */
  if(BLOCK_Y>=  4)if(ty==0)p_sh[0][ty][tx]+=p_sh[0][ty+ 3][tx];__syncthreads();
  if(BLOCK_Y>=  3)if(ty==0)p_sh[0][ty][tx]+=p_sh[0][ty+ 2][tx];__syncthreads();
  if(BLOCK_Y>=  2)if(ty==0)p_sh[0][ty][tx]+=p_sh[0][ty+ 1][tx];__syncthreads();

  if(BLOCK_X>=128)if(tx<64)p_sh[0][ty][tx]+=p_sh[0][ty][tx+64];__syncthreads();
  if(BLOCK_X>= 64)if(tx<32)p_sh[0][ty][tx]+=p_sh[0][ty][tx+32];__syncthreads();
  if (tx < 16) {
      p_sh[0][ty][tx]+=p_sh[0][ty][tx+16];__syncthreads();
      p_sh[0][ty][tx]+=p_sh[0][ty][tx+ 8];__syncthreads();
      p_sh[0][ty][tx]+=p_sh[0][ty][tx+ 4];__syncthreads();
      p_sh[0][ty][tx]+=p_sh[0][ty][tx+ 2];__syncthreads();
      p_sh[0][ty][tx]+=p_sh[0][ty][tx+ 1];__syncthreads();
  }
#if !defined(SWAP_XY)
  if ((tx | ty) != 0) gosa_o[blockIdx.y*gridDim.x+blockIdx.x] = p_sh[0][0][0];
#else
  if ((tx | ty) != 0) gosa_o[blockIdx.x*gridDim.y+blockIdx.y] = p_sh[0][0][0];
#endif /* SWAP_XY */
}


__global__ void
my_jacobi_kernel_btm_even (const float* RESTRICT a0d, const float* RESTRICT a1d, 
                        const float* RESTRICT a2d, const float* RESTRICT a3d,
                        const float* RESTRICT b0d, const float* RESTRICT b1d,
                        const float* RESTRICT b2d, const float* RESTRICT c0d,
                        const float* RESTRICT c1d, const float* RESTRICT c2d,
                        const float* RESTRICT wrk1,const float* RESTRICT bnd,
                        const float* RESTRICT p, float* RESTRICT wrk2,
                        float* RESTRICT gosa_o, float omega)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
#if !defined(SWAP_XY)
  int gx = tx + blockIdx.x*BLOCK_X;
  int gy = ty + blockIdx.y*BLOCK_Y+1;
#else
  int gx = tx + blockIdx.y*BLOCK_X;
  int gy = ty + blockIdx.x*BLOCK_Y+1;
#endif
  int index = gx + rsize*gy;
  //  int btm=0,mid=1,top;
  float a0,a1,a2,a3,b0,b1,b2,c0,c1,c2,s0,ss,work1,bound,gosa=0.0f;
  //__shared__ float p_sh[3][(BLOCK_Y+2)][(BLOCK_X+2)];
  __shared__ float p_sh[1][BLOCK_Y][BLOCK_X];
   

#if (BLOCK_Y < 2)
#error code requires BLOCK_Y >= 2
#endif
  /* Avoid accessing memory outside the array */
    /*  if ((index-rsize-1) >= 0) p_sh[btm][ty  ][tx  ] = fetch_p(index-rsize-1);
  p_sh[btm][ty  ][tx+2] = fetch_p(index-rsize+1);
  p_sh[btm][ty+2][tx  ] = fetch_p(index+rsize-1);
  p_sh[btm][ty+2][tx+2] = fetch_p(index+rsize+1);
  p_sh[mid][ty  ][tx  ] = fetch_p(index-rsize-1+psize);
  p_sh[mid][ty  ][tx+2] = fetch_p(index-rsize+1+psize);
  p_sh[mid][ty+2][tx  ] = fetch_p(index+rsize-1+psize);
  p_sh[mid][ty+2][tx+2] = fetch_p(index+rsize+1+psize);*/

  for(int z=1; z<MIDPLANE_EVEN; z++)
  {
    /*   top = (z+1)%3;
    mid = (z  )%3;
    btm = (z-1)%3;*/
    index += psize;
    /*__syncthreads();
    //load next plane
    p_sh[top][ty  ][tx  ] = fetch_p(index+psize-rsize-1);
    p_sh[top][ty  ][tx+2] = fetch_p(index+psize-rsize+1);
    p_sh[top][ty+2][tx  ] = fetch_p(index+psize+rsize-1);
    p_sh[top][ty+2][tx+2] = fetch_p(index+psize+rsize+1);
    __syncthreads();*/

    //apply stencil
    a0 = a0d[index];
    a1 = a1d[index];

    work1 = wrk1[index];
#if 0
    s0 = work1 + a0*p_sh[top][ty+1][tx+1] + a1*p_sh[mid][ty+2][tx+1];
    b0 = b0d[index];
    a2 = a2d[index];

    s0 += b0*( p_sh[top][ty+2][tx+1] - p_sh[top][ty][tx+1] - p_sh[btm][ty+2][tx+1] + p_sh[btm][ty][tx+1] ) + a2*p_sh[mid][ty+1][tx+2];
    b1 = b1d[index];
    s0 += b1*( p_sh[mid][ty+2][tx+2] - p_sh[mid][ty][tx+2] - p_sh[mid][ty+2][tx] + p_sh[mid][ty][tx] );
    b2 = b2d[index];
    c0 = c0d[index];
    s0 += b2*( p_sh[top][ty+1][tx+2] - p_sh[btm][ty+1][tx+2] - p_sh[top][ty+1][tx] + p_sh[btm][ty+1][tx] ) + c0*p_sh[btm][ty+1][tx+1];
    c1 = c1d[index];
    c2 = c2d[index];

    s0 += c1*p_sh[mid][ty][tx+1] + c2*p_sh[mid][ty+1][tx];
    a3 = a3d[index];
    bound = bnd[index];
#if defined(NEW_GOSA)
    ss = __fmul_rn(bound,__fadd_rn(__fmul_rn(s0,a3),-p_sh[mid][ty+1][tx+1]));
#else
    ss = bound*(s0*a3 - p_sh[mid][ty+1][tx+1]);
#endif

#else
    s0 = work1 + a0*fetch_p(index+psize) /*p_sh[top][ty+1][tx+1]*/ + a1*fetch_p(index + rsize) /*p_sh[mid][ty+2][tx+1]*/;
    b0 = b0d[index];
    a2 = a2d[index];
    s0 += b0*( fetch_p(index + psize + rsize)/*p_sh[top][ty+2][tx+1]*/ -  fetch_p(index + psize - rsize) /*p_sh[top][ty][tx+1]*/ - 
               fetch_p(index - psize + rsize)/*p_sh[btm][ty+2][tx+1]*/ +  fetch_p(index-psize - rsize)/*p_sh[btm][ty][tx+1]*/ ) + a2* fetch_p(index + 1)/*p_sh[mid][ty+1][tx+2]*/;
    b1 = b1d[index];
    s0 += b1*(  fetch_p(index + rsize + 1)/*p_sh[mid][ty+2][tx+2]*/ -  fetch_p(index - rsize + 1)/*p_sh[mid][ty][tx+2]*/ -  fetch_p(index + rsize -1) /*p_sh[mid][ty+2][tx]*/ +  fetch_p(index - rsize  - 1)/*p_sh[mid][ty][tx]*/ );
    b2 = b2d[index];
    c0 = c0d[index];
    s0 += b2*(  fetch_p(index+psize + 1)/*p_sh[top][ty+1][tx+2]*/ -  fetch_p(index - psize + 1)/*p_sh[btm][ty+1][tx+2]*/ -  fetch_p(index + psize - 1)/*p_sh[top][ty+1][tx]*/ +  fetch_p(index-psize - 1)/*p_sh[btm][ty+1][tx]*/ ) + 
      c0* fetch_p(index - psize)/*p_sh[btm][ty+1][tx+1]*/;
    c1 = c1d[index];
    c2 = c2d[index];

    s0 += c1* fetch_p(index - rsize )/*p_sh[mid][ty][tx+1]*/ + c2* fetch_p(index - 1)/*p_sh[mid][ty+1][tx]*/;
    a3 = a3d[index];
    bound = bnd[index];
#if defined(NEW_GOSA)
    ss = __fmul_rn(bound,__fadd_rn(__fmul_rn(s0,a3),- fetch_p(index )/*p_sh[mid][ty+1][tx+1]*/));
#else
    ss = bound*(s0*a3 -  fetch_p(index)/*p_sh[mid][ty+1][tx+1]*/);
#endif
#endif

    //write result
    if(gx<MKMAX-3&&gy<MJMAX-2)
    {
      wrk2[index] =  fetch_p(index)/*p_sh[mid][ty+1][tx+1]*/ + omega*ss;
      gosa += ss*ss;
    }
  }//end loop over z
  // Wait with p_sh update until all threads have finished reading p_sh
  __syncthreads();
  p_sh[0][ty][tx] = gosa;


  int id = threadIdx.x + threadIdx.y * BLOCK_X;
  reduce_block<float, (BLOCK_X * BLOCK_Y)>(id, &p_sh[0][ty][tx]);
  /*
  // Once all threads have written gosa to p_sh, reduce gosa in shared memory
  __syncthreads();

#if (BLOCK_X > 128) || (BLOCK_X < 32) || ((BLOCK_X & (BLOCK_X - 1)) != 0)
#error code requires that BLOCK_X is a power of 2, and 32 <= BLOCK_X <= 128
#endif
#if (BLOCK_Y > 4) 
#error code requires that BLOCK_Y is <= 4
#endif*/
  /*
  if(BLOCK_Y>=  4)if(ty==0)p_sh[0][ty][tx]+=p_sh[0][ty+ 3][tx];__syncthreads();
  if(BLOCK_Y>=  3)if(ty==0)p_sh[0][ty][tx]+=p_sh[0][ty+ 2][tx];__syncthreads();
  if(BLOCK_Y>=  2)if(ty==0)p_sh[0][ty][tx]+=p_sh[0][ty+ 1][tx];__syncthreads();

  if(BLOCK_X>=128)if(tx<64)p_sh[0][ty][tx]+=p_sh[0][ty][tx+64];__syncthreads();
  if(BLOCK_X>= 64)if(tx<32)p_sh[0][ty][tx]+=p_sh[0][ty][tx+32];__syncthreads();
  if (tx < 16) {
      p_sh[0][ty][tx]+=p_sh[0][ty][tx+16];__syncthreads();
      p_sh[0][ty][tx]+=p_sh[0][ty][tx+ 8];__syncthreads();
      p_sh[0][ty][tx]+=p_sh[0][ty][tx+ 4];__syncthreads();
      p_sh[0][ty][tx]+=p_sh[0][ty][tx+ 2];__syncthreads();
      p_sh[0][ty][tx]+=p_sh[0][ty][tx+ 1];__syncthreads();
      }*/


#if !defined(SWAP_XY)
  if ((tx | ty) != 0) gosa_o[blockIdx.y*gridDim.x+blockIdx.x] = p_sh[0][0][0];
#else
  if ((tx | ty) != 0) gosa_o[blockIdx.x*gridDim.y+blockIdx.y] = p_sh[0][0][0];
#endif 


}


void jacobi_GPU_btm_even (cudaStream_t stream, float *a0_d, float *a1_d, 
                          float *a2_d, float *a3_d, float *b0_d, float *b1_d,
                          float *b2_d, float *c0_d, float *c1_d, float *c2_d,
                          float *wrk1_d, float *bnd_d, float *p_d, 
                          float *wrk2_d, float *gosa_d, float omega, int n)
{
    // execution configuration
    dim3 dimBlock(BLOCK_X,BLOCK_Y,1);
#if !defined(SWAP_XY)
    dim3 dimGrid( GRID_X, GRID_Y);
#else
    dim3 dimGrid( GRID_Y, GRID_X);
#endif
    dim3 dimBlock_sum_gosa(512,1,1);
    dim3 dimGrid_sum_gosa(1,1,1);
#if defined(USE_TEXTURE)
    size_t texOfs;
    cudaBindTexture (&texOfs, ptex, p_d);
    if (texOfs) {
        fprintf (stderr, "error in file '%s' in line %i : %s.\n",
                 __FILE__, __LINE__,
                 "texOfs != 0, violates program assumptions");
        exit (EXIT_FAILURE);
    }
#endif

    /*    jacobi_kernel_btm_even<<<dimGrid,dimBlock,0,stream>>> (a0_d,a1_d,a2_d,a3_d,
                                                           b0_d,b1_d,b2_d,c0_d,
                                                           c1_d,c2_d, wrk1_d,
                                                           bnd_d,p_d,wrk2_d,
                                                           gosa_d,omega);

    // CHECK_LAUNCH_ERROR();
    sum_gosa<<<dimGrid_sum_gosa,dimBlock_sum_gosa,0,stream>>> (gosa_d,
    GRID_X*GRID_Y); */

    my_jacobi_kernel_btm_even<<<dimGrid,dimBlock,0,stream>>> (a0_d,a1_d,a2_d,a3_d,
                                                           b0_d,b1_d,b2_d,c0_d,
                                                           c1_d,c2_d, wrk1_d,
                                                           bnd_d,p_d,wrk2_d,
                                                           gosa_d,omega);

    sum_gosa<<<dimGrid_sum_gosa,dimBlock_sum_gosa,0,stream>>> (gosa_d,
    GRID_X*GRID_Y);

    // CHECK_LAUNCH_ERROR();
}


#if (CUDART_VERSION >= 3000)
void set_kernel_cache_config (enum cudaFuncCache mode)
{
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(sum_gosa, mode));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(jacobi_kernel_btm_even, mode));
#ifdef MULTIGPU
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(jacobi_kernel_top_even, mode));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(jacobi_kernel_btm_odd, mode));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(jacobi_kernel_top_odd, mode));
#endif
}
#endif /* CUDART_VERSION >= 3000 */
