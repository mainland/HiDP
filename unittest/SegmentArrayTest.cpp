#include <stdio.h>
#include <stdlib.h>
#include "gtest/gtest.h"
#include "NeUtility.h"
/*using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestCase;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;*/

// Step 2. Use the TEST macro to define your tests.
//
// TEST has two parameters: the test case name and the test name.
// After using the macro, you should define your test logic between a
// pair of braces.  You can use a bunch of macros to indicate the
// success or failure of a test.  EXPECT_TRUE and EXPECT_EQ are
// examples of such macros.  For a complete list, see gtest.h.
//
// <TechnicalDetails>
//
// In Google Test, tests are grouped into test cases.  This is how we
// keep test code organized.  You should put logically related tests
// into the same test case.
//
// The test case name and the test name should both be valid C++
// identifiers.  And you should not use underscore (_) in the names.
//
// Google Test guarantees that each test you define is run exactly
// once, but it makes no guarantee on the order the tests are
// executed.  Therefore, you should write your tests in such a way
// that their results don't depend on their order.
//
// </TechnicalDetails>
#include "SegmentArray.h"

// Tests Factorial().
template <class T>
static void seqInitArray(SegmentArray<T> &a)
{
  for (int i = 0; i < a.getSize(); i++)
    a[i] = i;
}

TEST(SegmentArray, Initialize) {
  // This test is named "Negative", and belongs to the "FactorialTest"
  // test case.
  NeUseDevice(&gDevProp);
  CudppPlanFactory planPool;
  SegmentArray<int> a(&planPool, 20);
  SegmentArray<int> zeroArray(&planPool, 0);
  EXPECT_EQ(1, a.getNumSegments());
  EXPECT_EQ(20, a.getSegmentLength(0));
  EXPECT_EQ(0, a.getSegmentOffset(0));
  EXPECT_EQ(0, a.getSegmentIndex(0));
  EXPECT_EQ(false, a.isRecursiveAllDone());
  a.setNewSegmentAt(10);
  EXPECT_EQ(10, a.getSegmentLength(0));
  EXPECT_EQ(10, a.getSegmentLength(1));
  a.setNewSegmentAt(5);
  EXPECT_EQ(5, a.getSegmentLength(0));
  EXPECT_EQ(5, a.getSegmentLength(1));
  EXPECT_EQ(10, a.getSegmentLength(2));
}

TEST(SegmentArray, LargeArray) {
  // This test is named "Negative", and belongs to the "FactorialTest"
  // test case.
  NeUseDevice(&gDevProp);
  int bigsize = gDevProp.totalGlobalMem/(sizeof(int) * 30);
  bigsize <<= 1;
  CudppPlanFactory planPool;
  SegmentArray<int> big(&planPool, bigsize);
  SegmentArray<int> zeroArray(&planPool, 0);
  EXPECT_EQ(1, big.getNumSegments());
  EXPECT_EQ(big.getSize(), big.getSegmentLength(0));
  EXPECT_EQ(0, big.getSegmentOffset(0));
  EXPECT_EQ(0, big.getSegmentIndex(0));
  EXPECT_EQ(big.getSize(), big.getSegmentLength(0));
  EXPECT_EQ(false, big.isRecursiveAllDone());
  big.setNewSegmentAt(bigsize/2);
  EXPECT_EQ(2, big.getNumSegments());
  EXPECT_EQ(bigsize/2, big.getSegmentLength(0));
  EXPECT_EQ(bigsize/2, big.getSegmentLength(1));
  big.setNewSegmentAt(100);
  EXPECT_EQ(3, big.getNumSegments());
  EXPECT_EQ(100, big.getSegmentLength(0));
  EXPECT_EQ((bigsize/2-100), big.getSegmentLength(1));
  EXPECT_EQ((bigsize/2), big.getSegmentLength(2));
}



TEST(SegmentArray, RegularReverse) {
  CudppPlanFactory planPool;
  int arraySize = 32;
  SegmentArray<int> a(&planPool, arraySize);
  seqInitArray(a);
  a.reverse();
  EXPECT_EQ(a[0], arraySize-1);
  seqInitArray(a);
  a.bottop(); // 16
  a.reverse();
  EXPECT_EQ(a[0], arraySize/2-1);
  seqInitArray(a);
  a.bottop(); // 8
  a.reverse();
  EXPECT_EQ(a[0], arraySize/4-1);
  seqInitArray(a);
  a.bottop(); // 4
  a.reverse();
  EXPECT_EQ(a[0], arraySize/8-1);
  seqInitArray(a);
  a.bottop(); // 2
  a.reverse();
  EXPECT_EQ(a[0], arraySize/16-1);

  seqInitArray(a);
  a.bottop(); // 1
  a.reverse();
  // unchanged
  for (int i = 0; i != a.getSize(); i++)
    EXPECT_EQ(a[i], i);
}

TEST(SegmentArray, ReverseArray) {
  NeUseDevice(&gDevProp);
  CudppPlanFactory planPool;
  SegmentArray<int> a(&planPool, 20);
  SegmentArray<int> zeroArray(&planPool, 0);
  EXPECT_EQ(1, a.getNumSegments());
  EXPECT_EQ(20, a.getSegmentLength(0));
  EXPECT_EQ(0, a.getSegmentOffset(0));
  EXPECT_EQ(0, a.getSegmentIndex(0));
  EXPECT_EQ(false, a.isRecursiveAllDone());

}

#if 0

TEST(SegmentArray, LargeArray) {
  // This test is named "Negative", and belongs to the "FactorialTest"
  // test case.
  NeUseDevice(&gDevProp);
  int bigsize = gDevProp.totalGlobalMem/(sizeof(int) * 20);
  CudppPlanFactory planPool;
  SegmentArray<int> a(&planPool, bigsize);
  SegmentArray<int> zeroArray(&planPool, 0);
  EXPECT_EQ(1, a.getNumSegments());
  EXPECT_EQ(a.getSize(), a.getSegmentLength(0));
  EXPECT_EQ(0, a.getSegmentOffset(0));
  EXPECT_EQ(0, a.getSegmentIndex(0));
  EXPECT_EQ(a.getSize(), a.getSegmentLength(0));
  EXPECT_EQ(false, a.isRecursiveAllDone());
  a.setSegRecursiveDone(0);
  EXPECT_EQ(true, a.isRecursiveAllDone());
  a.getSegments()[100] = 1;
  a.getSegments()[200] = 1;
  a.getSegments()[1000] = 1;
  a.getSegments()[a.getSize() - 1] = 1;
  a.segmentsChanged();
  EXPECT_EQ(5, a.getNumSegments());
  a.getSegments()[a.getSize() - 2] = 1;
  a.segmentsChanged();
  EXPECT_EQ(6, a.getNumSegments());
}


TEST(SegmentArray, ChangeSegments) {
  CudppPlanFactory planPool;
  SegmentArray<int> a(&planPool, 20);
  a.getSegments()[9] = 1;
  a.segmentsChanged();
  EXPECT_EQ(a.getSegmentIndex(0), 0);
  EXPECT_EQ(a.getSegmentIndex(8), 0);
  EXPECT_EQ(a.getSegmentIndex(9), 1);
  EXPECT_EQ(a.getSegmentIndex(19), 1);
  EXPECT_EQ(a.getNumSegments(), 2);
  EXPECT_EQ(a.getSegmentLength(0), 9);
  EXPECT_EQ(a.getSegmentLength(1), 11);
  EXPECT_EQ(a.getSegmentOffset(0), 0);
  EXPECT_EQ(a.getSegmentOffset(1), 9);
}

TEST(SegmentArray, ChangeSegmentsLoop) {
  CudppPlanFactory planPool;
  SegmentArray<int> a(&planPool, 20);
  for (int i = 0; i != a.getSize(); i++)
    {
      a.getSegments()[i] = 1;
      a.segmentsChanged();
      int i_1 = i+1;
      EXPECT_EQ(a.getNumSegments(), i_1);
    }
}


TEST(SegmentArray, PackSegments) {
  CudppPlanFactory planPool;
  SegmentArray<int> a(&planPool, 30);
  for (int i = 0; i != a.getSize(); i++)
    {
      a[i] = i;
    }
  a.getSegments()[15] = 1;
  a.segmentsChanged();
  MirroredArray<uint> flag(30);
  memset(flag.getPtr(), 0, sizeof(uint) * flag.getSize());
  flag[0] = 1; flag[10] = 1; flag[15] = 1;
  flag[20] = 1; flag[21] = 1;  flag[27] = 1; flag[25] = 1;
  SegmentArray<int> packedArray(&planPool, 0);
  a.packByFlag(flag, packedArray);
  EXPECT_EQ(packedArray[2], 15);
  EXPECT_EQ(packedArray[3], 20);
  EXPECT_EQ(packedArray.getSegmentIndex(0), 0);
  EXPECT_EQ(packedArray.getSegmentIndex(1), 0);
  EXPECT_EQ(packedArray.getSegmentIndex(2), 1);
  EXPECT_EQ(packedArray.getSegmentLength(0), 2);
  EXPECT_EQ(packedArray.getSegmentLength(1), 5);
}


TEST(SegmentArray, PackSegmentsHasEmptySeg) {
  CudppPlanFactory planPool;
  SegmentArray<int> a(&planPool, 30);
  for (int i = 0; i != a.getSize(); i++)
    {
      a[i] = i;
    }
  a.getSegments()[10] = 1;
  a.getSegments()[15] = 1;
  a.getSegments()[20] = 1;
  a.segmentsChanged();
  MirroredArray<uint> flag(30);
  flag.setToZero();
  //  memset(flag.getPtr(), 0, sizeof(uint) * flag.getSize());
  // set up so that the 0th and 2nd segments have no flags
  flag[11] = 1; flag[12] = 1; flag[13] = 1;
  flag[20] = 1; flag[21] = 1;  flag[27] = 1; flag[25] = 1;
  SegmentArray<int> packedArray(&planPool, 0);
  a.packByFlag(flag, packedArray);
  EXPECT_EQ(packedArray.getParentArraySegMapping()[0], (uint)1);
  EXPECT_EQ(packedArray.getParentArraySegMapping()[1], (uint)3);
}


TEST(SegmentArray, SubSeq) {
 CudppPlanFactory planPool;
  SegmentArray<int> a(&planPool, 30);
  for (int i = 0; i != a.getSize(); i++)
    {
      a[i] = i;
    }
 
  MirroredArray<int> lows(1);
  MirroredArray<int> his(1);
  lows[0] = 5;
  his[0] = 10;
  SegmentArray<int> sub(&planPool);
  a.subseq(lows, his, sub);
  EXPECT_EQ(sub[0], 5);
  EXPECT_EQ(sub[1], 6);
  EXPECT_EQ(sub.getSegLength()[0], his[0]-lows[0]);

}
#endif
