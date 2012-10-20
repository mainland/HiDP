#include "HiArray.h"

TEST(HiArray, HiArray_resize) {
  // 1-d array
  HiArray<unsigned int, 1> array(20);
  array.randomize();
  array(0) = 1;
  array(1) = 2;
  array.resize(10);
  EXPECT_EQ(2, array(1));
  EXPECT_EQ(1, array(0));
  array.resize(30); 
  EXPECT_EQ(2, array(1));
  EXPECT_EQ(1, array(0));

  // 2-d array
  int sizes[] = {3, 2, 1};
  HiArray<int, 1> array2(sizes);
  array2(0,0,0) = 2;
  EXPECT_EQ(2, array2(0,0,0));
  //  array.resize(20);
}
