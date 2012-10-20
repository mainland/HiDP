#include "MirroredArray.h"
#include "ScopeProfile.h"
#include "NeUtility.h"
#include <cstdio>

extern std::vector<StepProfile> gProfile;
extern std::map<std::string, long> gProfileGroup;
cudaDeviceProp gDevProp;

typedef int SORT_T;
#include "bitonic_kernel.cu"

void reverse_kernel_w(SORT_T *gBuffer, int sort_size, int sub_size)
{
  cudaDeviceProp *devProp = &gDevProp;
  int outer_par = sort_size / sub_size;
  int numBlocks = 32;
  int blockSize = 256;
  int block_per_outer = numBlocks/outer_par  + 1;
  numBlocks = block_per_outer * outer_par;
  reverse_kernel<<<numBlocks, blockSize>>>(gBuffer, sort_size, sub_size, block_per_outer);
  CHECK_KERNEL_ERROR("reverse_warp");
}

void reverse_warp_w(SORT_T *gBuffer, int sort_size, int sub_size)
{
  cudaDeviceProp *devProp = &gDevProp;
  int size = sort_size/4;
  reverse_warp<<<128, 256>>>(gBuffer, sort_size, sub_size);
  CHECK_KERNEL_ERROR("reverse_warp");
}

void reverse_thread_w(SORT_T *gBuffer, int sort_size, int sub_size)
{
  cudaDeviceProp *devProp = &gDevProp;
  int size = sort_size/4;
  reverse_thread<<<128, 256>>>(gBuffer, sort_size, sub_size);
  CHECK_KERNEL_ERROR("reverse_thread");
}



void bitonic_kernel_w(SORT_T *gBuffer, int sort_size, int sub_size)
{
  cudaDeviceProp *devProp = &gDevProp;
  int outer_par = sort_size / sub_size;
  int numBlocks = 32;
  int blockSize = 256;
  int block_per_outer = numBlocks/outer_par  + 1;
  numBlocks = block_per_outer * outer_par;

  int index = sub_size;
  int sub_size_log  = 0;
  while (index >>= 1) ++sub_size_log;
  // printf("numblocks %d. blockSize %d.\n", numBlocks, blockSize);
  bitonic_kernel<<<numBlocks, blockSize>>>(gBuffer, sort_size, sub_size, block_per_outer, sub_size_log);
  CHECK_KERNEL_ERROR("bitonic_kernel");
}

void bitonic_warp_w(SORT_T *gBuffer, int sort_size, int sub_size)
{
  cudaDeviceProp *devProp = &gDevProp;
  int size = sort_size/2;
  //  bitonic_warp<<<KERNEL_SETUP(size)>>>(gBuffer, sort_size, sub_size);
  int index = sub_size;
  int sub_size_log  = 0;
  while (index >>= 1) ++sub_size_log;

  bitonic_warp<<<128, 256>>>(gBuffer, sort_size, sub_size, sub_size_log);
  CHECK_KERNEL_ERROR("bitonic_warp");
}

void bitonic_subwarp_w(SORT_T *gBuffer, int sort_size, int sub_size)
{
  cudaDeviceProp *devProp = &gDevProp;
  int size = sort_size/2;

  int index = sub_size;
  int sub_size_log  = 0;
  while (index >>= 1) ++sub_size_log;

  bitonic_subwarp<<<128, 256>>>(gBuffer, sort_size, sub_size, sub_size_log);
  CHECK_KERNEL_ERROR("bitonic_subwarp");
}

void bitonic_thread_w(SORT_T *gBuffer, int sort_size, int sub_size)
{
  cudaDeviceProp *devProp = &gDevProp;
  int size = sort_size/2;

  int index = sub_size;
  int sub_size_log  = 0;
  while (index >>= 1) ++sub_size_log;

  bitonic_thread<<<128, 256>>>(gBuffer, sort_size, sub_size, sub_size_log);
  CHECK_KERNEL_ERROR("bitonic_thread");
}



void reverse(SORT_T *gBuffer, int sort_size, int sub_size)
{
  // tuning branches
  int outer_par = sort_size / sub_size;
  if (outer_par <= 64) 
    reverse_kernel_w(gBuffer, sort_size, sub_size);
  else if (sub_size >= 32)
    reverse_warp_w(gBuffer, sort_size, sub_size);
  else 
    reverse_thread_w(gBuffer, sort_size, sub_size);
}

void bitonic_sort(SORT_T *gBuffer, int sort_size, int sub_size)
{
  // tuning branches
  int outer_par = sort_size / sub_size;
  if (outer_par <= 256) 
    bitonic_kernel_w(gBuffer, sort_size, sub_size);
  else if (sub_size >= 64)
    bitonic_warp_w(gBuffer, sort_size, sub_size);
  else if (sub_size >= 8)
    bitonic_subwarp_w(gBuffer, sort_size, sub_size);
  else 
    bitonic_thread_w(gBuffer, sort_size, sub_size);
}


void bitonic_block_w(SORT_T *gBuffer, int sort_size, int sub_size)
{
  cudaDeviceProp *devProp = &gDevProp;
  //  int numBlocks = sort_size/sub_size;
  if (sub_size == 2048)
    //    bitonic_block<2048, 11><<<128, sub_size/2>>>(gBuffer, sort_size);
    //    bitonic_block<2048, 11><<<128, 128>>>(gBuffer, sort_size);
        bitonic_block2<<<128, 256>>>(gBuffer, sort_size);
    //     bitonic_block2_unroll<<<128, 256>>>(gBuffer, sort_size);
  else if (sub_size == 1024)
    bitonic_block<1024, 10><<<128, sub_size/2>>>(gBuffer, sort_size);
  else if (sub_size == 512)
    bitonic_block<512, 9><<<128, sub_size/2>>>(gBuffer, sort_size);
  else if (sub_size == 256)
    bitonic_block<256, 8><<<128, sub_size/2>>>(gBuffer, sort_size);
  else if (sub_size == 128)
    bitonic_block<128, 7><<<128, sub_size/2>>>(gBuffer, sort_size);
  else if (sub_size == 64)
    bitonic_block<64, 6><<<128, sub_size/2>>>(gBuffer, sort_size);
  else if (sub_size == 32)
    bitonic_block<32, 5><<<128, sub_size/2>>>(gBuffer, sort_size);
  else
    assert(0);
  CHECK_KERNEL_ERROR("bitonic_block");
}

void bitonic_sort_block(SORT_T *gBuffer, int sort_size, int sub_size)
{
  // tuning branches
  bitonic_block_w(gBuffer, sort_size, sub_size);
}

void runTest(int argc, char **argv)
{
  NeUseDevice(&gDevProp);
  cudaFuncSetCacheConfig(bitonic_block2, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(bitonic_block2_unroll, cudaFuncCachePreferShared);
  //  cudaFuncSetCacheConfig(minMaxKernel, cudaFuncCachePreferL1);

  for (int i = 19; i < 24; i++)
    {
    int sort_size = (2<<i);
  //  int sort_size = (2<<10);
  MirroredArray<SORT_T> toSort(sort_size);
  SORT_T *buffer = toSort.getPtr();
  for (int i = 0; i != sort_size; i++)
    buffer[i] = (sort_size - i);//rand();
  //   buffer[i] = rand()/2;

  //  toSort.print("before reverse");  
  SORT_T *gBuffer = toSort.getGpuPtr();

  //  reverse(gBuffer, sort_size, 8);
  //  toSort.print("after reverse");
  //  return;

  ScopeProfile *total = new ScopeProfile("batchsort total");
  int outer_size = 2;
  //   toSort.print("init value");
  while (outer_size <= sort_size)
    {
      // reverse
      int inner_size = outer_size;
      if (outer_size >= 4)
        {
          reverse(toSort.getGpuPtr(), sort_size, outer_size);
          //                   toSort.print("after reversing");
        }

      while (inner_size >= 2)
        {
          if (inner_size <= 2048 && inner_size >= 256)
          //          if (inner_size == )
            {
              //                            toSort.print("before bulk sorting");
              bitonic_sort_block(toSort.getGpuPtr(), sort_size, inner_size);
              //              toSort.print("after bulk sorting");              
                          break;
            }
          else
            {
              bitonic_sort(toSort.getGpuPtr(), sort_size, inner_size);
              //              toSort.print("after sorting");
              inner_size>>=1 ;
            }
          
        }
      // 
      outer_size <<= 1;
      
    }
  
  delete total;
  if (!toSort.isFSorted(-1, -1))
    printf("Warning!!!!! array is not sorted.\n");

  //  toSort.print("after sorting");
  printf("sorting size %d.\n", sort_size);
    }
  showProfileResult(gProfile);

}
