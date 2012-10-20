#pragma once
#include <map>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

// TODO contains code generation part for a particular parallel function
// e.g. reduce has properties such as whether need allocate global memory (yes if at kernel level)
// whether need append another kernel (yes if at kernel level)
class Capability{
public:
    
private:
    bool mMapSuffix;  // whether it is a suffix to a map
    
};

/* A linked list for a hierarchy of execution models */
class HierarchyMode{
 public:
  explicit HierarchyMode(const char *name);
    virtual ~HierarchyMode() {}

    void push_back(boost::shared_ptr<HierarchyMode>);
  void setSuitableRange(int min, int max);
  void incNumBelow();
  void incNumUpper();
  void setNumUpper(int upper) ;
  void setNumBelow(int below) ;
  int getNumUpper() const { return mLevelUp; }
  int getNumBelow() const { return mLevelBelow; }
    
    int getSuitableMinRange() const { return mRangeMin; }
    int getSuitableMaxRange() const { return mRangeMax; }


  HierarchyMode* getUpper();
    boost::shared_ptr<HierarchyMode> getBelow();
    
    void setNext(boost::shared_ptr<HierarchyMode> next) { mBelow = next; }
    void setPrevious(HierarchyMode *prev) { mUpper = prev; }
  // TODO  void addCapability(const std::string &); 
    
    void addCapability();
 private:
  std::string mName;
  int mRangeMin;  // suitable range
  int mRangeMax;
  int mLevelBelow;
  int mLevelUp;
    boost::shared_ptr<HierarchyMode> mBelow; // only responsible for releaseing belowing models
    HierarchyMode * mUpper;

    // todo use set?

  std::map<std::string, Capability*> mCapabilities;  // contains the supported parallel functions at this level
    std::vector<boost::shared_ptr<HierarchyMode> > mChildren;
    struct HierarchyModel *mParent;
};
