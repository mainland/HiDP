#include "Dim.h"
#include "ParserNodes.h"

bool Dim::operator==(const Dim&rhs)
{
  if (getNumDims() != rhs.getNumDims())
      return false;
    for (unsigned int i = 0; i < getNumDims(); i++)
    {
        // TODO
    }
  assert(0);
  return true;
}


bool DimType::operator==(const DimType &rhs) const
{
    if (!isConst() && !rhs.isConst())
    {
        const VarDim *lhsp = dynamic_cast<const VarDim *>(this);
        const VarDim *rhsp = dynamic_cast<const VarDim *>(&rhs);    
        return ((lhsp->name == rhsp->name) && lhsp->blockId == rhsp->blockId);
    } else if (isConst() && rhs.isConst())
    {
        const IntDim *lhsp = dynamic_cast<const IntDim *>(this);
        const IntDim *rhsp = dynamic_cast<const IntDim *>(&rhs);    
        return (lhsp->val == rhsp->val);
    }
    else 
        return false;
}