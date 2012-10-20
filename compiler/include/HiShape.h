#pragma once


#include <vector>
#include <list>
#include <string>
#include <set>
#include <boost/shared_ptr.hpp>

class HiSymbol;
class ParserNode;
class DefIndexParserNode;


struct HiShapeItem{
    boost::shared_ptr<ParserNode> mLeftLimiter;
    boost::shared_ptr<ParserNode> mRightLimiter;
    boost::shared_ptr<HiSymbol> mSymbol;
} ;

// base class for HiShape
class BlockParserNode;
class HiShape{
public:

//    HiShape(int left, int right);
//    HiShape(int left, boost::shared_ptr<DefIndexParserNode> right);
    
    void appendShape(int left, int right);
    void appendShape(int left,  boost::shared_ptr<HiSymbol> symbol, boost::shared_ptr<ParserNode> right);
    void appendShape(BlockParserNode *block, const std::string &id, boost::shared_ptr<ParserNode> left, boost::shared_ptr<ParserNode> right);
    void appendShape(const HiShapeItem &item);
    void removeShape(std::list<HiShapeItem>::iterator iter);
//    bool isCompatible(const HiShape &rhs);
    bool operator==(const HiShape &rhs);
//  operator
    size_t getSize() const { return mShape.size(); }
    void print() const ;
    
    std::list<HiShapeItem>::iterator findShape(const std::string &name, bool &found);
    void emitToString(std::stringstream &) const;
    
    bool isCompatible(boost::shared_ptr<class HierarchyMode> model) const;
//    boost::shared_ptr<DefIndexParserNode> getLeft() { return mLeftLimiter; }
//    boost::shared_ptr<DefIndexParserNode> getRight() { return mLeftLimiter; }    
private:
    // 
//    std::vector<boost::shared_ptr<DefIndexParserNode> >mLeftLimiters;
//    std::vector<boost::shared_ptr<DefIndexParserNode> >mRightLimiters;
//      std::vector<boost::shared_ptr<ParserNode> >mLeftLimiters;  // TODO make it an expression node
//      std::vector<boost::shared_ptr<ParserNode> >mRightLimiters;
//    std::vector<boost::shared_ptr<HiSymbol> > mLeftSymbols;
//    std::vector<boost::shared_ptr<HiSymbol> > mRightSymols;
                     
    std::list<HiShapeItem> mShape;
    std::set<std::string> mPrimitives;
};



#if 0
// scale shape
class HiScalarShape: public HiShape{
public:
    HiScalarShape() : HiShape(){}
//    virtual bool compatibleWith
private:
    
};

class RangeParserNode;
// vector shape
class HiVectorShape : public HiShape{
public:
private:
    std::vector<RangeParserNode *> mVector;
};
#endif


class HiShapeLayer{
public:
    HiShapeLayer()
    {
        mLayers.clear();
    }
    void addLayer(HiShape layer)
    {
        mLayers.push_back(layer);
    }
    
    HiShapeLayer(const HiShapeLayer &rhs)
    {
        mLayers = rhs.mLayers;
    }
    bool isCompatible(const HiShapeLayer &rhs);
    
    void hprint(int level) const;
    void promoteShape(struct DefParserNode *range);
    
    void addShapeAtIndex(const HiShapeItem &item, int index);
    size_t getSize() const {return mLayers.size(); }
   
    size_t getNumLayers() const { return mLayers.size(); }
    
    std::string emitString() const;
    std::string emitFlatString() const;
    void fuseShape(const HiShapeLayer &shape);
    
    bool operator==(const HiShapeLayer &rhs) const;
    
    HiShape &getLayer(size_t index) { return mLayer[index]; }
private:
    // 
    std::vector<HiShape> mLayers;  // nested shapes
};


