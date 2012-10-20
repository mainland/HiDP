#include <stdio.h>
#include <stdlib.h>
#include "gtest/gtest.h"
#include "cuda_runtime_api.h"
cudaDeviceProp gDevProp;


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestCase;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

//#include "HiArrayTest.cpp"
#include "BlockSortTest.cpp"
//#include "PartitionTest.cpp"

/*#include "SubSegmentArrayTest.cpp"
#include "SegmentArrayTest.cpp"
#include "CudppPlanFactoryTest.cpp"
#include "MirroredArrayTest.cpp"*/
//#include "TestMiscKernels.cpp"

