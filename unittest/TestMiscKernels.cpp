#include <stdio.h>
#include <stdlib.h>
#include "gtest/gtest.h"

#include "miscKernels.h"
#include "MirroredArray.h"

TEST(MistKernels, threadwarpsync)
{
  MirroredArray<uint> a(200);
  testThreadSync(a.getGpuPtr(), a.getSize());
  for (int i = 0; i < a.getSize(); i++)
    {
      printf("%d ", a[i]);
    }
  printf("\n");
}
  // This test is named "Negative", and belongs to the "FactorialTest"

