#pragma once
#include "HierarchyMode.h"
#include <string>


// single thread model
class HiThreadModel : public HierarchyMode {
public:
    HiThreadModel():HierarchyMode("thread")
    {
        setSuitableRange(1024, INT_MAX);
    }
    
};

//
class HiSubWarpModel: public HierarchyMode {
public:
    HiSubWarpModel():HierarchyMode("subwarp"),
        mNumThreads(8)
    {
    setSuitableRange(256, 1024*4);
    }
    
private:
    int mNumThreads;
};

class HiSubWarp2Model: public HierarchyMode {
public:
    HiSubWarp2Model():HierarchyMode("subwarp2"),
    mNumThreads(4)
    {
        setSuitableRange(256, 1024*8);

    }
    
private:
    int mNumThreads;
};



class HiWarpModel: public HierarchyMode {
public:
    HiWarpModel():HierarchyMode("warp"),
    mNumThreads(32)
    {
        setSuitableRange(256, 1024);
    }
private:
    int mNumThreads;
};

class HiBlockModel: public HierarchyMode {
public:
    HiBlockModel():HierarchyMode("block")
    {
    setSuitableRange(16, 256);
    }
private:
    std::string mOneDShape; // blockDim.x
    std::string mTwoDShape; // blockDim.x and blockDim.y;
};

class HiKernelModel: public HierarchyMode {
public:
    HiKernelModel():HierarchyMode("kernel")
    {
        setSuitableRange(1, 16);
    }
private:
    
};