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
#include "SubSegmentArray.h"

// Tests Factorial().
template <class T>
static void seqInitArray(SegmentArray<T> &a)
{
  for (int i = 0; i < a.getSize(); i++)
    a[i] = i;
}

TEST(SubSegmentArray, Botsegreverse) {
  // This test is named "Negative", and belongs to the "FactorialTest"
  // test case.
  NeUseDevice(&gDevProp);
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
  }
}

