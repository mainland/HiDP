#pragma once

#include "HiSymbol.h"


class HiSymbolTree{
public:
    bool operator==(const HiSymbolTree &rhs) const;
private:
    boost::shared_ptr<HiSymbol> mSymbol;    // if mSymbol exist, it is a leaf
    
    // otherwise, it has two children
    std::string mOp;
    boost::shared_ptr<HiSymbolTree> mLhs;
    boost::shared_ptr<HiSymbolTree> mRhs;
};