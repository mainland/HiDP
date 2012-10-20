#include <stdio.h>
#include <stdlib.h>
#include "gtest/gtest.h"
#include "NeUtility.h"

#include "MirroredArray.h"
// Tests Factorial().
extern cudaDeviceProp gDevProp;
// Tests factorial of negative numbers.
TEST(MirroredArray, Copy) {
   MirroredArray<uint> flag(30);
   MirroredArray<uint> a_flag(flag);
   MirroredArray<uint> b_flag(10);
   b_flag = flag;
}
