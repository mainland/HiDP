#include "HiSymbolTree.h"


bool HiSymbolTree::operator==(const HiSymbolTree &rhs) const
{
    if (mSymbol && rhs.mSymbol)
        return mSymbol == rhs.mSymbol;
    else if (!mSymbol)
    {
        if (!rhs.mSymbol)
        {
            if (mOp != rhs.mOp)
                return false;
            return ((mLhs == rhs.mLhs) && (mRhs == rhs.mRhs));
        }
        else
            return false;
    }
    return false;
        
}