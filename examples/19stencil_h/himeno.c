#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"

#include "himeno.h"

#undef ALLOC_DEBUG
#define USE_PAD
#define PRINT_ITER  (10)

#if defined(ALLOC_DEBUG)
#define DBGMSG(args) (printf("DEBUG: "), printf args)
#else
#define DBGMSG(args)
#endif

// Local functions
float jacobi(int);
void check_results(void);
void initmt(void);
void allocate_memory(void);
void cleanup(void);


double wallclock(void);

static int imax, jmax, kmax, imax_global, ME, NP, gpu;
static float omega;

static float *a0;
static float *a1;
static float *c1;
static float *c2;

static float *gosa_btm;
static float *gosa_top;
static float *a2;
static float *a3;
static float *b0;
static float *b1;
static float *b2;
static float *c0;
static float *bnd;
static float *wrk;
static float *p1;
static float *p2;

static float *a0_d;
static float *a1_d;
static float *c1_d;
static float *c2_d;

static float *a2_d;
static float *a3_d;
static float *b0_d;
static float *b1_d;
static float *b2_d;
static float *c0_d;
static float *wrk_d;
static float *bnd_d;
static float *p1_d;
static float *p2_d;
static float *gosa_d;

static float *a0_d_orig;
static float *a1_d_orig;
static float *c1_d_orig;
static float *c2_d_orig;

static float *a2_d_orig;
static float *a3_d_orig;
static float *b0_d_orig;
static float *b1_d_orig;
static float *b2_d_orig;
static float *c0_d_orig;
static float *wrk_d_orig;
static float *bnd_d_orig;
static float *p1_d_orig;
static float *p2_d_orig;
static float *gosa_d_orig;

static int memsize = PITCH*MJMAX*MIMAX*sizeof(float);

int main( int argc, char*argv[] )
{
  float *p_old,*p_new,*p_tmp;
  int    n,nn;
  float  gosa,gflops,thruput,thruput2;
  double time_start,time_max,target,bytes;
  cudaStream_t stream_top,stream_btm;

  NP=1;
  gpu=0;
  ME=0;

  target= 60.0;
  omega= 0.8f;
  imax = MIMAX-1;
  jmax = MJMAX-1;
  kmax = MKMAX-1;
  imax_global = NP*(imax-2)+2;
  nn = ITERS;

  if(ME==0)
  {
    printf("\n mimax = %d mjmax = %d mkmax = %d pitch = %d\n",MIMAX, MJMAX, MKMAX, PITCH);
    printf(" imax = %d jmax = %d kmax = %d\n",imax_global,jmax,kmax);
    printf(" gridX = %d  gridY = %d  blockX = %d  blockY = %d\n", GRID_X, GRID_Y, BLOCK_X, BLOCK_Y);
  }
  //printf("There are %d processes, I am process# %d using GPU %d\n",NP,ME,gpu);
  
  CUDA_SAFE_CALL(cudaSetDevice(gpu));
  stream_top = 0; 
  stream_btm = 0;

#if (CUDART_VERSION >= 3000)
  {
#if (CUDART_VERSION > 3000)
      struct cudaDeviceProp prop;
      // display ECC configuration, only queryable post r3.0
      CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, gpu));
      printf (" ECC on GPU %d is %s\n", gpu, prop.ECCEnabled ? "ON" : "OFF");
#endif /* CUDART_VERSION > 3000 */
      // configure kernels for large shared memory to get better occupancy
      printf (" Configuring GPU L1 cache size ...\n");
      set_kernel_cache_config (cudaFuncCachePreferShared);
  }
#endif /* CUDART_VERSION >= 3000 */

  CUDA_SAFE_CALL(cudaStreamCreate(&stream_top));
  CUDA_SAFE_CALL(cudaStreamCreate(&stream_btm));

  if(ME==0) printf(" Allocating Memory...\n");
  allocate_memory();
  if(ME==0) printf(" Initializing Data...\n\n");
  initmt();
  
  if(ME==0)
  {
    printf(" Now, start GPU measurement process.\n");
    printf(" The loop will be excuted %d times\n",nn);
    printf(" Wait for a while\n\n");
  }

  time_start = wallclock();

  gosa = 0.0f;
  p_new = p2_d; p_old = p1_d;
  
  for(n=0 ; n<nn; n++)
  {
    //swap pointers
    p_tmp = p_new; p_new = p_old; p_old = p_tmp;
    jacobi_GPU_btm_even (stream_btm,a0_d,a1_d,a2_d,a3_d,b0_d,b1_d,b2_d,c0_d,
                         c1_d,c2_d,wrk_d,bnd_d,p_old,p_new,gosa_d,omega,n);

    cudaMemcpyAsync (gosa_btm, gosa_d, sizeof(float), cudaMemcpyDeviceToHost,
                     stream_btm);
    // Since we want to print intermediate values of gosa every PRINT_ITER
    // iterations, we need to synchronize before picking up the asynchronously 
    // updated value.
    if (!(n % PRINT_ITER)) {
        cudaStreamSynchronize(stream_btm);
        gosa = *gosa_btm;
    }
    if(ME==0 && n%PRINT_ITER==0) printf(" iter: %d \tgosa: %e\n",n,gosa);
  }

  cudaThreadSynchronize();
  gosa = *gosa_btm;
  time_max = wallclock() - time_start;

  gflops   = (float)(34.0*( (double)nn*(double)(imax_global-2)*(double)(jmax-2)*(double)(kmax-2) ) / time_max * 1e-9);
  bytes    = NP*((double)nn*(56.0*(imax-2)+8.0)*(double)(jmax)*(double)(kmax));
  thruput  = (float)(bytes / time_max / 1024.0 / 1024.0 / 1024.0);
  thruput2 = (float)(bytes / time_max / 1e9);

  if(ME==0)
  {
    printf(" \nLoop executed for %d times\n",nn);
    printf(" Gosa : %e \n",gosa);
    printf(" total Compute   : %4.1f GFLOPS\ttime : %f seconds\n",gflops,time_max);
    printf(" total Bandwidth : %4.1f GB/s\n", thruput);
    printf(" total Bandwidth : %4.1f GB/s (STREAM equivalent)\n",thruput2);
    printf(" Score based on Pentium III 600MHz : %f\n\n",1000.0*gflops/82.0);
  }
  cleanup();

  CUDA_SAFE_CALL(cudaStreamDestroy(stream_top));
  CUDA_SAFE_CALL(cudaStreamDestroy(stream_btm));

  //check_results();
  return (EXIT_SUCCESS);
}

void initmt(void)
{
    int i,j,k;
    int ii;
    int cpysize = memsize - sizeof(float);
  
    for (i=0 ; i<MIMAX ; i++) {
        for (j=0 ; j<MJMAX ; j++) {
            for (k=0 ; k<MKMAX ; k++) {
                a0 __(i,j,k)=0.0f;
                a1 __(i,j,k)=0.0f;
                c1 __(i,j,k)=0.0f;
                c2 __(i,j,k)=0.0f;

                a2 __(i,j,k)=0.0f;
                a3 __(i,j,k)=0.0f;
                b0 __(i,j,k)=0.0f;
                b1 __(i,j,k)=0.0f;
                b2 __(i,j,k)=0.0f;
                c0 __(i,j,k)=0.0f;
                p1 __(i,j,k)=0.0f;
                p2 __(i,j,k)=0.0f;
                wrk __(i,j,k)=0.0f;
                bnd __(i,j,k)=0.0f;
            }
        }
    }

    for (i=0 ; i<imax ; i++) {
        for (j=0 ; j<jmax ; j++) {
            for (k=0 ; k<kmax ; k++) {
                ii = i + (imax-2)*ME;
                a0 __(i,j,k)=1.0f;
                a1 __(i,j,k)=1.0f;
                c1 __(i,j,k)=1.0f;
                c2 __(i,j,k)=1.0f;

                a2 __(i,j,k)=1.0f;
                a3 __(i,j,k)=1.0f/6.0f;
                b0 __(i,j,k)=0.0f;
                b1 __(i,j,k)=0.0f;
                b2 __(i,j,k)=0.0f;
                c0 __(i,j,k)=1.0f;
                p1 __(i,j,k)=(float)(ii*ii)/((imax_global-1)*(imax_global-1));
                p2 __(i,j,k)=p1 __(i,j,k);
                wrk __(i,j,k)=0.0f;
                bnd __(i,j,k)=1.0f;
            }
        }
    }

    CUDA_SAFE_CALL(cudaMemcpy(a0_d, a0+1, cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(a1_d, a1+1, cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(c1_d, c1+1, cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(c2_d, c2+1, cpysize, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(a2_d, a2+1, cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(a3_d, a3+1, cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(b0_d, b0+1, cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(b1_d, b1+1, cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(b2_d, b2+1, cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(c0_d, c0+1, cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(wrk_d,wrk+1,cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bnd_d,bnd+1,cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(p1_d, p1+1, cpysize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(p2_d, p2+1, cpysize, cudaMemcpyHostToDevice));
}

void check_results(void)
{
      //cudaMemcpy(p2, p2_d, memsize,cudaMemcpyHostToDevice);
      int i,j,k;
      float maxerr = 0.f;
      int maxi=0,maxj=0,maxk=0;
      int err=0;
      for(i=1 ; i<imax-1 ; i++)
      for(j=1 ; j<jmax-1 ; j++){
        for(k=1 ; k<kmax-1 ; k++){
            float gold = p1 __(i,j,k);
            float mine = p2 __(i,j,k);
            float perr = 100.f*(float)fabs(gold-mine)/gold;
            if(perr>maxerr)
            {
              maxerr = perr;
              maxi = i;
              maxj = j;
              maxk = k;
            }
            if(perr>0.1f) {
            err = 1;
            printf("%f percent error at a[%d][%d][%d] = CPU: %g , GPU: %g\n",perr,i,j,k,gold,mine);
            getchar();
            }
         }
       }
  printf("max percent error = %f at a[%d][%d][%d] = CPU: %g , GPU: %g\n",maxerr,maxi,maxj,maxk,p1 __(maxi,maxj,maxk),p2 __(maxi,maxj,maxk));
  if(err==0) printf("PASSED ....................................  \n");
}

float jacobi(int nn)
{
  int i,j,k,n;
  float s0, ss;
  double gosa = 0.0;
  
  for(n=0 ; n<nn ; ++n){

    gosa = 0.0f;

    for(i=1 ; i<imax-1 ; i++)
      for(j=1 ; j<jmax-1 ; j++)
        for(k=1 ; k<kmax-1 ; k++){
          s0 = a0 __(i,j,k) * p2 __(i+1,j,k)
             + a1 __(i,j,k) * p2 __(i,j+1,k)
             + a2 __(i,j,k) * p2 __(i,j,k+1)
             + b0 __(i,j,k) * ( p2 __(i+1,j+1,k) - p2 __(i+1,j-1,k)
                              - p2 __(i-1,j+1,k) + p2 __(i-1,j-1,k) )
             + b1 __(i,j,k) * ( p2 __(i,j+1,k+1) - p2 __(i,j-1,k+1)
                              - p2 __(i,j+1,k-1) + p2 __(i,j-1,k-1) )
             + b2 __(i,j,k) * ( p2 __(i+1,j,k+1) - p2 __(i-1,j,k+1)
                              - p2 __(i+1,j,k-1) + p2 __(i-1,j,k-1) )
             + c0 __(i,j,k) * p2 __(i-1,j,k)
             + c1 __(i,j,k) * p2 __(i,j-1,k)
             + c2 __(i,j,k) * p2 __(i,j,k-1)
             + wrk __(i,j,k);

          ss = ( s0 * a3 __(i,j,k) - p2 __(i,j,k) ) * bnd __(i,j,k);

          gosa += (double)ss*(double)ss;

          /* gosa= (gosa > ss*ss) ? a : b; */

          p1 __(i,j,k) = p2 __(i,j,k) + omega * ss;
        }

    for(i=1 ; i<imax-1 ; ++i)
      for(j=1 ; j<jmax-1 ; ++j)
        for(k=1 ; k<kmax-1 ; ++k)
          p2 __(i,j,k) = p1 __(i,j,k);
  } /* end n loop */

  return((float)gosa);
}

void allocate_memory(void)
{
    //allocate host arrays
    a0 = (float*)malloc(memsize);
    a1 = (float*)malloc(memsize);
    c1 = (float*)malloc(memsize);
    c2 = (float*)malloc(memsize);

    a2 = (float*)malloc(memsize);
    a3 = (float*)malloc(memsize);
    b0 = (float*)malloc(memsize);
    b1 = (float*)malloc(memsize);
    b2 = (float*)malloc(memsize);
    c0 = (float*)malloc(memsize);
    wrk = (float*)malloc(memsize);
    bnd = (float*)malloc(memsize);
    if (!a0 || !a1 || !a2 || !a3 || !b0 || !b1 || !b2 || 
        !c0 || !c1 || !c2 || !wrk || !bnd) 
    {
        fprintf(stderr, "Host allocation error in file '%s' in line %i\n",
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    
    //allocate pressure array page-locked
    CUDA_SAFE_CALL(cudaMallocHost((void**)&p1,memsize));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&p2,memsize));

    //allocate page-locked gosa variables
    CUDA_SAFE_CALL(cudaMallocHost((void**)&gosa_btm,sizeof(float)));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&gosa_top,sizeof(float)));
    
#if defined(USE_PAD)
    // This padding & offsetting is a workaround for a problem with the r3.0
    // global memory allocator
#define PAD    (1024*1024)
#define OFS    (6*1024)
#else
#define PAD    (0)
#define OFS    (0)
#endif /* defined(USE_PAD) */
    
    //allocate device arrays
    CUDA_SAFE_CALL(cudaMalloc((void**)&gosa_d_orig, 
                              GRID_X * GRID_Y * sizeof(gosa_d[0])));
    gosa_d = 0*OFS + gosa_d_orig; 
    DBGMSG(("gosa_d = %10p  size = %10lu\n", 
            gosa_d, GRID_X * GRID_Y * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&a0_d_orig,memsize + PAD));
    a0_d   = 1*OFS + a0_d_orig;
    DBGMSG (("a0_d   = %10p  size = %10d  pad = %10d\n", a0_d,  memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&a1_d_orig,memsize + PAD));
    a1_d   = 2*OFS + a1_d_orig;
    DBGMSG (("a1_d   = %10p  size = %10d  pad = %10d\n", a1_d,  memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&c1_d_orig,memsize + PAD));
    c1_d   = 9*OFS + c1_d_orig;
    DBGMSG (("c1_d   = %10p  size = %10d  pad = %10d\n", c1_d,  memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&c2_d_orig,memsize + PAD));
    c2_d   = 10*OFS + c2_d_orig;
    DBGMSG (("c2_d   = %10p  size = %10d  pad = %10d\n", c2_d,  memsize, PAD));

    CUDA_SAFE_CALL(cudaMalloc((void**)&a2_d_orig,memsize + PAD));
    a2_d   = 3*OFS + a2_d_orig;
    DBGMSG (("a2_d   = %10p  size = %10d  pad = %10d\n", a2_d,  memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&a3_d_orig,memsize + PAD));
    a3_d   = 4*OFS + a3_d_orig;
    DBGMSG (("a3_d   = %10p  size = %10d  pad = %10d\n", a3_d,  memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&b0_d_orig,memsize + PAD));
    b0_d   = 5*OFS + b0_d_orig;
    DBGMSG (("b0_d   = %10p  size = %10d  pad = %10d\n", b0_d,  memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&b1_d_orig,memsize + PAD));
    b1_d   = 6*OFS + b1_d_orig;
    DBGMSG (("b1_d   = %10p  size = %10d  pad = %10d\n", b1_d,  memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&b2_d_orig,memsize + PAD));
    b2_d   = 7*OFS + b2_d_orig;
    DBGMSG (("b2_d   = %10p  size = %10d  pad = %10d\n", b2_d,  memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&c0_d_orig,memsize + PAD));
    c0_d   = 8*OFS + c0_d_orig;
    DBGMSG (("c0_d   = %10p  size = %10d  pad = %10d\n", c0_d,  memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&wrk_d_orig,memsize + PAD));
    wrk_d  = 11*OFS + wrk_d_orig;
    DBGMSG (("wrk_d  = %10p  size = %10d  pad = %10d\n", wrk_d, memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&bnd_d_orig,memsize + PAD));
    bnd_d  = 12*OFS + bnd_d_orig;
    DBGMSG (("bnd_d  = %10p  size = %10d  pad = %10d\n", bnd_d, memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&p1_d_orig,memsize + PAD));
    p1_d   = 13*OFS + p1_d_orig;
    DBGMSG (("p1_d   = %10p  size = %10d  pad = %10d\n", p1_d,  memsize, PAD));
    CUDA_SAFE_CALL(cudaMalloc((void**)&p2_d_orig,memsize + PAD));
    p2_d   = 14*OFS + p2_d_orig;
    DBGMSG (("p2_d   = %10p  size = %10d  pad = %10d\n", p2_d,  memsize, PAD));
}

void cleanup(void)
{
    //FREE host arrays
    free(a0);
    free(a1);
    free(c1);
    free(c2);

    free(a2);
    free(a3);
    free(b0);
    free(b1);
    free(b2);
    free(c0);
    free(wrk);
    free(bnd);
  
    //FREE pressure array page-locked
    CUDA_SAFE_CALL(cudaFreeHost(p1));
    CUDA_SAFE_CALL(cudaFreeHost(p2));
  
    //FREE gosa
    CUDA_SAFE_CALL(cudaFreeHost(gosa_btm));
    CUDA_SAFE_CALL(cudaFreeHost(gosa_top));

    //FREE device arrays
    CUDA_SAFE_CALL(cudaFree(gosa_d_orig));
    CUDA_SAFE_CALL(cudaFree(a0_d_orig));  
    CUDA_SAFE_CALL(cudaFree(a1_d_orig));
    CUDA_SAFE_CALL(cudaFree(c1_d_orig));
    CUDA_SAFE_CALL(cudaFree(c2_d_orig));

    CUDA_SAFE_CALL(cudaFree(a2_d_orig));
    CUDA_SAFE_CALL(cudaFree(a3_d_orig));
    CUDA_SAFE_CALL(cudaFree(b0_d_orig));
    CUDA_SAFE_CALL(cudaFree(b1_d_orig));
    CUDA_SAFE_CALL(cudaFree(b2_d_orig));
    CUDA_SAFE_CALL(cudaFree(c0_d_orig));
    CUDA_SAFE_CALL(cudaFree(wrk_d_orig));
    CUDA_SAFE_CALL(cudaFree(bnd_d_orig));
    CUDA_SAFE_CALL(cudaFree(p1_d_orig));
    CUDA_SAFE_CALL(cudaFree(p2_d_orig));
}

#ifndef MULTIGPU
#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
double wallclock (void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency (&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter (&t);
        return (double)t.QuadPart * oofreq;
    } else {
        return (double)GetTickCount() / 1000.0;
    }
}
#elif defined(__linux__) || defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
double wallclock (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif
#endif
