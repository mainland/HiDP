#pragma once

#include "HierachyMode.h"
#include <boost/shared_ptr.hpp>
#include <vector>

// a tree structure for all possible mapping
class MappingTree{
public:
    MappingTree()
    {
        
    }
    
    
private:
    std::vector<boost::shared_ptr<HierachyMode> > mMapping;
    
};