#include "cuda_runtime.h"

#undef USE_FLOAT2
#define USE_M_CONFIG

#if defined(USE_M_CONFIG)
// This is the "M" configuration
#define MIMAX            129
#define MJMAX            129
#define MKMAX            257
#else
// This is the "L" configuration
#define MIMAX            257
#define MJMAX            257
#define MKMAX            513
#endif

#define ITERS (100)
#define SPLIT (50)

#ifdef MULTIGPU
#define MIDPLANE_EVEN (SPLIT*(MIMAX-1)/100)
#define MIDPLANE_ODD  ((MIMAX-1)-MIDPLANE_EVEN)
#else
#define MIDPLANE_EVEN (MIMAX-2)
#endif

// These are the dimensions of the tile which we'll use to cover each plane.
// The code assumes that each tile element is mapped to one thread, so these
// are also the dimensions of the thread block (CTA) that operates on a tile.
// In general we BLOCK_X should be a multiple of 32 so we get fully coalesced
// GMEM accesses across a complete warp of 32 threads. Experimentally we find
// that a 64x2 configuration gives the best performance on Tesla-architecture 
// GPUs, but that a 64x3 configuration is the best for Fermi-architecture GPUs.
#if 1
#define BLOCK_X (64)
#define BLOCK_Y (3)
#else

#define BLOCK_X (64)
#if defined(FERMI)
#define BLOCK_Y (3)
#else
#define BLOCK_Y (2)
#endif

#endif

// Compute the CTA grid size needed to cover the active region of each plane,
// assuming we have a 1:1 correspondence between threads and plane elements.
// Each plane of MJMAX by MKMAX elements is surrounded by a one-element halo 
// for a total of 2 halo elements in each dimension, plus there is a pad of 1
// element per dimension. So the dimensions of the active region are smaller
// by three compared to the allocated plane.
#define GRID_X  (((MKMAX-2-1)+(BLOCK_X-1))/BLOCK_X)
#define GRID_Y  (((MJMAX-2-1)+(BLOCK_Y-1))/BLOCK_Y)

// We expect MKMAX to be larger than 64 and of the form 2^n+1, i.e. of size 2^n
// plus 1 for padding. Since we strip the padding in the x-dimension, subtract
// the 1, so the element pitch PITCH is of the form 2^n and > 64. This ensures
// that the byte pitch is a multiple of the 256-byte alignment needed for fully
// coalesced GMEM accesses.
#define PITCH   (MKMAX-1)
#if (PITCH < 64) || ((PITCH & (PITCH - 1)) != 0)
#error code requires that PITCH is a power of 2 that is >= 64
#endif

#define rsize   (PITCH)
#define psize   ((PITCH)*MJMAX)
#define ofs     (PITCH*MJMAX*MIMAX)
#define __(i, j, k) [(k)+(j)*(PITCH)+(i)*(MJMAX)*(PITCH)]

#define RESTRICT __restrict__

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)


// Functions exported from himeno.cuda.cu
#if defined(__cplusplus)
extern "C" {
#endif
#ifdef USE_FLOAT2
void jacobi_GPU_btm_even (cudaStream_t stream, float2* a0_a1_d, 
                          float* a2_d, float* a3_d, float* b0_d, float* b1_d,
                          float* b2_d, float* c0_d, float2* c1_c2_d,
                          float* wrk1_d, float* bnd_d, float* p_d, 
                          float* wrk2_d, float* gosa_d, float omega, int n);
#else
void jacobi_GPU_btm_even (cudaStream_t stream, float* a0_d, float* a1_d, 
                          float* a2_d, float* a3_d, float* b0_d, float* b1_d,
                          float* b2_d, float* c0_d, float* c1_d, float* c2_d,
                          float* wrk1_d, float* bnd_d, float* p_d, 
                          float* wrk2_d, float* gosa_d, float omega, int n);
#endif
#ifdef MULTIGPU
void jacobi_GPU_top_even (cudaStream_t stream, float* a0_d, float* a1_d,
                          float* a2_d, float* a3_d, float* b0_d, float* b1_d,
                          float* b2_d, float* c0_d, float* c1_d, float* c2_d,
                          float* wrk1_d, float* bnd_d, float* p_d, 
                          float* wrk2_d, float* gosa_d, float omega, int n);
void jacobi_GPU_btm_odd (cudaStream_t stream, float* a0_d, float* a1_d, 
                         float* a2_d, float* a3_d, float* b0_d, float* b1_d,
                         float* b2_d, float* c0_d, float* c1_d, float* c2_d,
                         float* wrk1_d, float* bnd_d, float* p_d,
                         float* wrk2_d, float* gosa_d, float omega, int n);
void jacobi_GPU_top_odd (cudaStream_t stream, float* a0_d, float* a1_d, 
                         float* a2_d, float* a3_d, float* b0_d, float* b1_d,
                         float* b2_d, float* c0_d, float* c1_d, float* c2_d,
                         float* wrk1_d, float* bnd_d, float* p_d,
                         float* wrk2_d, float* gosa_d, float omega, int n);
#endif

#if (CUDART_VERSION >= 3000)
void set_kernel_cache_config (enum cudaFuncCache mode);
#endif

#if defined(__cplusplus)
}
#endif
