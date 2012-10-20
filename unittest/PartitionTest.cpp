
typedef unsigned int PartitionType ;

struct MyCompare
{
__host__ __device__ 
int operator()(const PartitionType &x, const PartitionType &pivot)  const
  { 
    if (x < pivot)
      return 0;
    else if (x > pivot)
      return 1;
    else return 2;
  }
};

#include "HiArray.h"
#include "KernelConfig.h"
#include "partition.cuh"

typedef unsigned int PartitionType ;

class PartitionTest : public testing::Test {
 protected:
  virtual void SetUp()
  {
    SetUpScale(); 
    array.resize(s_size);
    array2.resize(s_size);
    pivots.resize(num_blocks);

    low.resize(num_blocks);
    high.resize(num_blocks);

    new_low.resize(num_blocks*2);
    new_high.resize(num_blocks*2);

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
     partition_kernel_wrapper<PartitionType, MyCompare>(num_blocks, array, array2, pivots, low, high, new_low, new_high);
    
  }
  virtual void SetUpInput() = 0 ;

  virtual void SetUpScale() 
  {
    setSize(2<<20);
    setSeg(3);
  }

  void setSize(int size)
  {
    s_size = size;
  }
  void setSeg(int seg)
  {
    num_blocks = seg;
  }

  void testCommon()
  {
  const PartitionType *ptr = array2.getReadOnlyPtr();
  const  PartitionType *pivot = pivots.getReadOnlyPtr();
  //  new_low.print();
  //  new_high.print();
  for (int b = 0; b != num_blocks; b++)
    {
      int less_begin = new_low(2*b);
      int less_end = new_high(2*b);
      int greater_begin = new_low(2*b+1);
      int greater_end = new_high(2*b+1);
      if (b > 0)
        EXPECT_LE(less_end, greater_begin);

      for (int s = less_begin; s < less_end; s++)
        {
          EXPECT_LE(ptr[s], pivot[b]);
        }
      for (int s = greater_begin; s < greater_end; s++)
        {
          EXPECT_GE(ptr[s], pivot[b]);
          if (ptr[s] < pivot[b])
            printf("wrong order at %d. %d < %d.\n", s, ptr[s], pivot[b]);
        }
    }

  }
  HiArray<PartitionType, 1> array;
  HiArray<PartitionType, 1> array2;
  HiArray<int, 1> low;
  HiArray<int, 1> high;
  HiArray<int, 1> new_low;
  HiArray<int, 1> new_high;
  HiArray<PartitionType, 1> pivots;

  int s_size;
  int num_blocks;
  time_t start_time_;
};

class PartitionTestRandom : public  PartitionTest{
  virtual void SetUpInput()
  {
    array.randomize();
    //    array.sequence(0);
    
    //pivots(0) = s_size/2;
    pivots.randomize();
  }
};


class PartitionTestSimple : public  PartitionTest{
  virtual void SetUpInput()
  {
    array.sequence(1);
    pivots(0) = s_size/2;
    /*    int chunk_size = s_size/num_blocks;
    for (int i = 0; i != num_blocks; i++)
    pivots(i) = array(i*chunk_size + chunk_size/2);*/
    //      pivots.randomize();
  }

  virtual void SetUpScale()
  {
    setSize(2<<10);
    setSeg(1);
  }
};


class PartitionTestAllZero : public  PartitionTest{
  virtual void SetUpInput()
  {
    array.setToZero();
    pivots(0) = 0;
    /*    int chunk_size = s_size/num_blocks;
    for (int i = 0; i != num_blocks; i++)
    pivots(i) = array(i*chunk_size + chunk_size/2);*/
    //      pivots.randomize();
  }

  virtual void SetUpScale()
  {
    setSize(2<<20);
    setSeg(1);
  }
};


class PartitionTestSequence : public  PartitionTest{
  virtual void SetUpInput()
  {
    array.sequence(1);
    int chunk_size = s_size/num_blocks;
    for (int i = 0; i != num_blocks; i++)
      pivots(i) = array(i*chunk_size + chunk_size/2);
    //      pivots.randomize();
  }

  virtual void SetUpScale()
  {
    setSize((2<<20) + 10);
    setSeg(16);
  }
};

TEST_F(PartitionTestSequence, testOneSequence)
{
  testCommon();
}

TEST_F(PartitionTestSimple, testOneSimple)
{
  testCommon();
  EXPECT_EQ(new_low(0), 0);
  EXPECT_EQ(new_low(1), 1024);
  EXPECT_EQ(new_high(0), 1023);
  EXPECT_EQ(new_high(1), 2048);
  //  printf("new begin: %d %d %d %d.\n", new
}

TEST_F(PartitionTestAllZero, testAllZero)
{
  testCommon();
  EXPECT_EQ(true, array2.isAllZero());
  EXPECT_EQ(new_low(0), 0);
  EXPECT_EQ(new_low(1), s_size);
  EXPECT_EQ(new_high(0), 0);
  EXPECT_EQ(new_high(1), new_low(1));
  //  printf("new begin: %d %d %d %d.\n", new
}

TEST_F(PartitionTestRandom, testOneRandom)
{
  testCommon();
  /*
  const PartitionType *ptr = array2.getReadOnlyPtr();
  const  PartitionType *pivot = pivots.getReadOnlyPtr();
  //  new_low.print();
  //  new_high.print();
  for (int b = 0; b != num_blocks; b++)
    {
      int less_begin = new_low(2*b);
      int less_end = new_high(2*b);
      int greater_begin = new_low(2*b+1);
      int greater_end = new_high(2*b+1);
      for (int s = less_begin; s < less_end; s++)
        {
          EXPECT_LE(ptr[s], pivot[b]);
        }
      for (int s = greater_begin; s < greater_end; s++)
        {
          EXPECT_GE(ptr[s], pivot[b]);
          if (ptr[s] < pivot[b])
            printf("wrong order at %d. %d < %d.\n", s, ptr[s], pivot[b]);
        }
        }*/
}


/*
class BlockSortTestRevSequence : public  BlockSortTest{
  virtual void SetUpInput()
  {
    array.sequence(1);
  }
  };*/


