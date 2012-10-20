#pragma once
#include "global.h"
#include "HierarchyMode.h"
#include <boost/shared_ptr.hpp>

void  config_gpu_execution_mode(const Params &param, boost::shared_ptr<HierarchyMode> &root);
//void  config_gpu_execution_mode(const Params &param, boost::shared_ptr<HierarchyMode> &root);
//void  free_gpu_execution_mode(HierarchyMode **root);

