#include "HiArray.h"
#include "ScopeProfile.h"
#include "NeUtility.h"
#include <cstdio>
#include <iostream>
#include <sstream>
#include "Operator.h"
#include "reduce.cuh"
#include "partition.cuh"
#include "sort.cuh"
#include "copy.cuh"
#include "KernelConfig.h"
#include "dists.h"
extern std::vector<StepProfile> gProfile;
extern std::map<std::string, long> gProfileGroup;
cudaDeviceProp gDevProp;

using namespace std;
typedef unsigned int SORT_T;
//typedef HiPair SORT_T;
#include "quicksort_kernel.cu"

#define DEBUG 0
#define DISTS 1

__global__ void HiArray_map_0(SORT_T *mPivot,  const SORT_T *mMin, const SORT_T *mMax,int num_task)
{
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int stepsize = blockDim.x * gridDim.x;
  for (int i = gid; i < num_task; i+= stepsize)
    {
      mPivot[i] = (mMin[i] + mMax[i])/2;
      //      printf("pivot is %d.\n", mPivot[i]);
    }
}

__global__ void HiArray_map_1(int *start, int *end, const int *begin, const int *last, int num_task)
{
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int stepsize = blockDim.x * gridDim.x;
  for (int i = gid; i < num_task; i+= stepsize)
    {
      start[i] = last[i*2];
      end[i] = begin[i*2+1];
    }
}

struct MyCompare
{
__host__ __device__ 
int operator()(const SORT_T &x, const SORT_T &pivot)  const
  { 
    if (x < pivot)
      return 0;
    else if (x > pivot)
      return 1;
    else return 2;
  }
};



template <class T>
class QuickSort{
 public:
  explicit QuickSort(int size, int t):
  mSize(size),
  type(t),
  max_task(64),
  iter(0),
  mBuffer(size),
  mOutput(size),
  mMin(max_task),
  mMax(max_task),
  mPivot(max_task),
  taskBegin1(max_task),
  taskEnd1(max_task),
  taskBegin2(max_task),
  taskEnd2(max_task)
  {
    setupInput();
  }
  
  void setupInput()
  {
    SORT_T *buffer = mBuffer.getPtr();
#if DISTS
    unsigned int *intermediate = new unsigned int[mSize];
    dist(intermediate, mSize, type);
    for (int i =0 ; i != mSize; i++)
      buffer[i] = intermediate[i];
    delete intermediate;
#else
    for (int i = 0; i != mSize; i++)
      //y
      buffer[i] = (mSize - i);//rand();
    //      buffer[i] = rand();
#endif
    //    mBuffer.setToZero();
    /*    for (int i = 0; i != mSize; i++)
        //       
        buffer[i] = rand();*/
    taskBegin1(0) = 0;
    taskEnd1(0) = mSize;
      //   buffer[i] = rand()/2;
  }

  void sort()
  {
    //    string title = string("sorting ") + string(itoa(mSize));
    ostringstream title;
    title << "sorting " << mSize;
    ScopeProfile total(title.str().c_str());
    //    T *input = mBuffer.getGpuPtr();
    //    T *output = mOutput.getGpuPtr();
    //    const int max_task = 32;
    int num_task = 1;
    iter = 0;
    HiArray<int, 1> *begin, *end, *new_begin, *new_end;
   HiArray<T, 1> *input, *output;

     while (num_task < max_task)
      {
        if ((iter % 2) == 0)
          {
            begin = &taskBegin1; end = &taskEnd1; new_begin = &taskBegin2; new_end = &taskEnd2;  input = &mBuffer; output = &mOutput;
          }
        else
          {
            begin = &taskBegin2; end = &taskEnd2; new_begin = &taskBegin1; new_end = &taskEnd1; input = &mOutput; output = &mBuffer;
          }
        //        printf("***** gen min *** \n");
        reduce_kernel_wrapper<MinOP<SORT_T>, SORT_T >(num_task, *input, mMin, *begin, *end);
        //        CHECK_KERNEL_ERROR("reduce(min) kernel");
        //        printf("%d.\n", mMin(0).key);
        //        printf("***** gen max *** \n");
        reduce_kernel_wrapper<MaxOP<SORT_T>, SORT_T >(num_task, *input, mMax, *begin, *end);
        
        HiArray_map_0<<<GRID_SIZE_COMMON, BLOCK_SIZE_COMMON>>>(mPivot.getGpuPtr(), mMin.getReadOnlyGpuPtr(), mMax.getReadOnlyGpuPtr(), num_task);

#if DEBUG
        printf("min and max pivot:\n");
        for (int i = 0; i != num_task; i++)
          printf("(%d %d %d) ", mMin(i).key, mMax(i).key, mPivot(i).key);
        printf("\n");
#endif

        partition_kernel_wrapper<SORT_T, MyCompare>(num_task, *input, *output, mPivot, *begin,  *end, *new_begin, *new_end);

#if DEBUG
        printf("new begin end:\n");
        for (int i = 0; i != num_task; i++)
          printf("(%d %d) (%d %d)", ((*new_begin)(2*i)), ((*new_end)(2*i)), ((*new_begin)(2*i+1)), ((*new_end)(2*i+1)));
        printf("\n");
#endif
        //        printf("new begin end (%d %d) (%d %d)\n", (*new_begin)(0), (*new_end)(0), (*new_begin)(1), (*new_end)(0));
        //  mPivot 
        if ((iter % 2) == 1)
          {
            HiArray<int , 1> copy_begin(num_task), copy_end(num_task);

            HiArray_map_1<<<GRID_SIZE_COMMON, BLOCK_SIZE_COMMON>>>(copy_begin.getGpuPtr(), copy_end.getGpuPtr(), (*new_begin).getReadOnlyGpuPtr(), (*new_end).getReadOnlyGpuPtr(), num_task);
#if DEBUG
            printf("copy begin and copy end:\n");
            copy_begin.print();
            copy_end.print();
#endif

            copy_block_wrapper<SORT_T>(num_task, *output, *input, copy_begin, copy_end);
          }

        iter++;
        num_task <<= 1;
      }
     // swap in and out
     {
       HiArray<T, 1> *tmp  = input;
       input = output;
       output = tmp;
     }
#if DEBUG
     (*input).saveToFile("before.txt");
     printf("printing new begin /end\n");
     (*new_begin).print();
     (*new_end).print();
#endif
     sort_block_wrapper<SORT_T>(num_task, *input, *output, *new_begin, *new_end);
     CHECK_KERNEL_ERROR("block sort kernel");
#if DEBUG
     //     (*output).saveToFile("after.txt");
#endif
  }
  void verify()
  {
    bool isSorted ;
    if ((iter & 1) == 0)
      isSorted = mOutput.isFSorted();
    else
      isSorted = mBuffer.isFSorted();

    if (!isSorted)
      cout << "Warning!!! array is not sorted!!!!!! " << endl;
    //    cout << "array is all zero? " << mOutput.isAllZero() << endl;
  }
 private:
  int mSize;
  int type;
  int max_task; 
  int iter;
HiArray<T,1 > mBuffer;
HiArray<T,1> mOutput;
HiArray<T, 1> mMin;
HiArray<T, 1> mMax;
HiArray<T, 1> mPivot;

HiArray<int, 1> taskBegin1;
HiArray<int, 1> taskEnd1;
HiArray<int, 1> taskBegin2;
HiArray<int, 1> taskEnd2;
  

};

void runTest(int argc, char **argv)
{
  NeUseDevice(&gDevProp);
  //  cudaFuncSetCacheConfig(minMaxKernel, cudaFuncCachePreferL1);
  if (argc != 2)
    {
      printf("please indicate distribution type. exiting...\n");
      return;
    }
  int type = atoi(argv[1]);
      int begin = 19; int end = 25;
    //    int begin = 17; int end = 17;
      //      int begin = 22; int end = 22;
  for (int i = begin; i <= end; i++)
    {
      int sort_size = (2<<i);
      printf("sorting size %d.\n", sort_size);
      //  int sort_size = (2<<10);
      QuickSort<SORT_T> qs(sort_size, type);
      qs.sort();
      qs.verify();

      
      //  toSort.print("before reverse");  
      //      SORT_T *gBuffer = toSort.getGpuPtr();
      
    //  reverse(gBuffer, sort_size, 8);
    //  toSort.print("after reverse");
    //  return;


  
    //    if (!toSort.isFSorted())
    //      printf("Warning!!!!! array is not sorted.\n");
    
    //  toSort.print("after sorting");

    }
  showProfileResult(gProfile);
}
