
#include "CudppPlanFactory.h"

TEST(CudppPlanFactory, Initialize) {
  CudppPlanFactory planPool;

  CUDPPConfiguration config;
  config.op = CUDPP_ADD;
  config.datatype = CUDPP_INT;
  config.algorithm = CUDPP_SCAN;
  config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

  CUDPPHandle handle1 =  planPool.getCudppPlan(config, 20, 1, 0);
  CUDPPHandle handle2 =  planPool.getCudppPlan(config, 10, 1, 0);
  CUDPPHandle handle3 =  planPool.getCudppPlan(config, 30, 1, 0);
  config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
  CUDPPHandle handle4 =  planPool.getCudppPlan(config, 30, 1, 0);

  config.algorithm = CUDPP_REDUCE;
  CUDPPHandle handle5 =  planPool.getCudppPlan(config, 30, 1, 0);

  EXPECT_EQ(handle1, handle2);
  //  EXPECT_NE(handle1, handle3);
  EXPECT_NE(handle1, handle4);
  EXPECT_NE(handle2, handle4);
  EXPECT_NE(handle3, handle4);

  EXPECT_NE(handle1, handle5);
  EXPECT_NE(handle2, handle5);
  EXPECT_NE(handle3, handle5);
  EXPECT_NE(handle4, handle5);
}
