#include "HierarchyMode.h"
using namespace std;
using namespace boost;
HierarchyMode::HierarchyMode(const char *name):
  mName(name),
  mRangeMin(0),  // suitable range
  mRangeMax(INT_MAX),
  mLevelBelow(0),
  mLevelUp(0),
mUpper(NULL),
mParent(NULL)
{
  
}


void HierarchyMode::push_back(shared_ptr<HierarchyMode> nextLevel)
{
  mBelow = nextLevel;
  nextLevel->setNumUpper(mLevelUp + 1);
  nextLevel->setPrevious(this);
  HierarchyMode * upper = mUpper;
  incNumBelow();
  while (upper)
    {
      upper->incNumBelow();
      upper = upper->getUpper();
    }
}

void HierarchyMode::setSuitableRange(int min, int max)
{
  mRangeMin = min;
  mRangeMax = max;
}

void HierarchyMode::incNumBelow()
{
  mLevelBelow++;
}

void HierarchyMode::incNumUpper()
{
  mLevelUp++;
}

void HierarchyMode::setNumBelow(int below)
{
  mLevelBelow = below;
}

void HierarchyMode::setNumUpper(int upper)
{
  mLevelUp = upper;
}


HierarchyMode *HierarchyMode::getUpper()
{
  return mUpper;
}

shared_ptr<HierarchyMode> HierarchyMode::getBelow()
{
  return mBelow;
}


/*void HierarchyMode::addCapability(const string &supported)
{
  mCapability.insert(supported);
  }*/
