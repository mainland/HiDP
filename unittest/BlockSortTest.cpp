#include "HiArray.h"
#include "KernelConfig.h"
#include "sort.cuh"
#include "HiTypes.h"
// auto generated one


//typedef unsigned int BlockSortType;
typedef HiPair BlockSortType;
//typedef int BlockSortType;
//typedef blocksor

__global__ void blocksort(BlockSortType *input, BlockSortType *output, 
                          const int *low, const int *high, int num_segs)
{
  sort_block<BlockSortType, BLOCK_SIZE_SORT>(input, output, low, high, num_segs);
  //each block sort input from low to high 
}

class BlockSortTest : public testing::Test {
 protected:
  virtual void SetUp()
  {
    SetUpScale(); 
    array.resize(s_size);
    array2.resize(s_size);

    low.resize(num_blocks);
    high.resize(num_blocks);
    for (int i = 0; i != num_blocks; i++)
      {
        low(i) = i * (s_size/num_blocks);
        if (i != (num_blocks-1))
          high(i) = (i+1)*(s_size/num_blocks);
      }
    high(num_blocks-1) = s_size;
    //    low.print();
    //    high.print();

    SetUpInput();
    start_time_ = time(NULL);
    blocksort<<<num_blocks, BLOCK_SIZE_SORT >>>(array.getGpuPtr(), array2.getGpuPtr(), 
                                                low.getReadOnlyGpuPtr(), high.getReadOnlyGpuPtr(),
                                                num_blocks);
  }
  virtual void SetUpInput() = 0 ;

  virtual void SetUpScale() 
  {
    setSize(2<<20);
    setSeg(1);
  }

  void setSize(int size)
  {
    s_size = size;
  }
  void setSeg(int seg)
  {
    num_blocks = seg;
  }

  HiArray<BlockSortType, 1> array;
  HiArray<BlockSortType, 1> array2;
  HiArray<int, 1> low;
  HiArray<int, 1> high;
  int s_size;
  int num_blocks;
  time_t start_time_;
};


class BlockSortTestSmall : public  BlockSortTest{
  virtual void SetUpScale()
  {
    setSeg(32);
    setSize(2<<7);
  }
  virtual void SetUpInput()
  {
    array.sequence(0);
  }
};


class BlockSortTestRevSequence : public  BlockSortTest{
  virtual void SetUpInput()
  {
    array.sequence(1);
  }
};

class BlockSortTestSequence : public  BlockSortTest{
  virtual void SetUpInput()
  {
    array.sequence(0);
  }
};

class BlockSortTestMultiSequence : public  BlockSortTest{
  virtual void SetUpScale()
  {
    setSeg(64);
    setSize(2<<22);
  }
  virtual void SetUpInput()
  {
    array.sequence(0);
  }
};

class BlockSortTestMultiRevSequence : public  BlockSortTest{
  virtual void SetUpScale()
  {
    //    setSeg(32);
    //    setSize(2<<23);
    setSeg(1);
    setSize(2<<8);
  }
  virtual void SetUpInput()
  {
    array.sequence(1);

  }
  virtual void TearDown() {
    //    const time_t end_time = time(NULL);
    //    std::cout << "sorting " << s_size  << " in " <<  num_blocks << " segments took " << end_time - start_time_ << " seconds." << std::endl;
  }
};


/*TEST_F(BlockSortTestSmall, testSmall)
{
  EXPECT_EQ(true, array2.isFSorted());
  const BlockSortType *ptr = array2.getReadOnlyPtr();
  for (int s = 0; s != s_size; s++)
    EXPECT_EQ(ptr[s], (s));
    }*/


TEST_F(BlockSortTestRevSequence, testOneRevSeq)
{
  EXPECT_EQ(true, array2.isFSorted());
  const BlockSortType *ptr = array2.getReadOnlyPtr();
  for (int s = 0; s != s_size; s++)
    EXPECT_EQ(ptr[s], (s+1));
}

TEST_F(BlockSortTestSequence, testOneSeq)
{
  EXPECT_EQ(true, array2.isFSorted());
  const BlockSortType *ptr = array2.getReadOnlyPtr();
  for (int s = 0; s != s_size; s++)
    EXPECT_EQ(ptr[s], (s));
}

TEST_F(BlockSortTestMultiSequence, testMultiSeq)
{
  const BlockSortType *ptr = array2.getReadOnlyPtr();
  for (int s = 0; s != s_size; s++)
    EXPECT_EQ(ptr[s], (s));
}


TEST_F(BlockSortTestMultiRevSequence, testMultiRevSeq)
{
  //  const BlockSortType *ptr = array2.getReadOnlyPtr();
  for (int s = 0; s != num_blocks; s++)
    {
      EXPECT_EQ(true, array2.isFSorted(low(s), high(s)));
    }
}

#if 0
TEST(BlockSort, BlockSortSimpleSequence) {
  static int num_blocks = 1;
  // This test is named "Negative", and belongs to the "FactorialTest"
  // test case.
  HiArray<BlockSortType, 1> array;
  HiArray<BlockSortType, 1> array2;
  HiArray<int, 1> low;
  HiArray<int, 1> high;
  int s_size = 2<<10;
    {
      array.resize(s_size);
      array2.resize(s_size);
      array.sequence(1);
      low.resize(num_blocks);
      high.resize(num_blocks);
      for (int i = 0; i != num_blocks; i++)
        {
          low(i) = i * (s_size/num_blocks);
          if (i != (num_blocks-1))
            high(i) = (i+1)*(s_size/num_blocks);
        }
      high(num_blocks-1) = s_size;
      
      blocksort<<<num_blocks, BLOCK_SIZE_SORT >>>(array.getGpuPtr(), array2.getGpuPtr(), 
                                                  low.getReadOnlyGpuPtr(), high.getReadOnlyGpuPtr(),
                                                  num_blocks);

      EXPECT_EQ(true, array2.isFSorted());
      const BlockSortType *ptr = array2.getReadOnlyPtr();
      for (int s = 0; s != s_size; s++)
        EXPECT_EQ(ptr[s], (s+1));
    }
}


TEST(BlockSort, BlockSortMultiple) {
  static int num_blocks = 1;
  // This test is named "Negative", and belongs to the "FactorialTest"
  // test case.
#if 1
  HiArray<BlockSortType, 1> array;
  HiArray<BlockSortType, 1> array2;
  HiArray<int, 1> low;
  HiArray<int, 1> high;
  int start = 2<<10;
  int end = 2<<11;
  for (int s_size = start; s_size < end; s_size<<=1)
    {
      array.resize(s_size);
      array2.resize(s_size);
#if 1
      array.randomize();
#else
      array.sequence(1);
#endif
      low.resize(num_blocks);
      high.resize(num_blocks);
      for (int i = 0; i != num_blocks; i++)
        {
          low(i) = i * (s_size/num_blocks);
          if (i != (num_blocks-1))
            high(i) = (i+1)*(s_size/num_blocks);
        }
      high(num_blocks-1) = s_size;
      
      /*      low.print();
              high.print();*/
      blocksort<<<num_blocks, BLOCK_SIZE_SORT >>>(array.getGpuPtr(), array2.getGpuPtr(), 
                                                  low.getReadOnlyGpuPtr(), high.getReadOnlyGpuPtr(),
                                                  num_blocks);

      //      array2.print();
      EXPECT_EQ(true, array2.isFSorted());

    }
#endif  

  /*  NeUseDevice(&gDevProp);
  CudppPlanFactory planPool;
  SegmentArray<int> a(&planPool, 32);
  seqInitArray(a);
  SubSegmentArray<int> bota(&planPool, &a, Sub_Bot);
  bota.reverse();
  EXPECT_EQ(31, a[16]);  
  {
    SegmentArray<int> a(&planPool, 30);
    seqInitArray(a);
    a.bottop();
    SubSegmentArray<int> bota(&planPool, &a, Sub_Bot);
    //    EXPECT_EQ(2, bota.getNumSegments());
    bota.reverse();
    EXPECT_EQ(14, a[7]);  
    }*/
}

#endif
