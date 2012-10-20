#pragma once
#include "utility.h"

#include <map>
#include <vector>
#include <iostream>
#include <boost/shared_ptr.hpp>

/*ntypedef enum
{
    HiSymbolConst,        // constant or define, no need to generate code
    // from function definition header, no need to generate definition code.
    HiSymbolFuncInput,    
    HiSymbolFuncOutput,
    HiSymbolFuncInout,     
    HiSymbolIndex,        // from array index, need to generate codels
    HiSymbolMapLocal    // from an undeclared variable in map block, need to declare it for legal code generation
} HiSymbolSource;
*/

class HiShape;
class HiSymbol;
class HiSymbolDimension;
class ParserNode;
class IndexDescription {
public:
    IndexDescription(boost::shared_ptr<HiSymbol> src, const ParserNode *node, int index) :
    mSource(src),
    mIndexNode(node),
    mDimIndex(index)
    {
        
    }
//private:
    boost::shared_ptr<HiSymbol> mSource;  // the array symbol 
//    boost::shared_ptr<HiSymbolDimension> mRef; // the array dimension symbol
    const ParserNode *mIndexNode;
    int mDimIndex;
};



class HiSymbol{
public:
/*    HiSymbol(std::string str, std::string type)://, int dim, int blockId):
    mSymbolName(str),
    mSymbolType(type),
    mSymbolDim(0),
    mBlockId(0),
    mDeclared(false)
    {
        
    }*/
    
    HiSymbol(std::string str, std::string type, int dim, int blockId):
    mSymbolName(str),
    mSymbolType(type),
    mSymbolDim(dim),
    mBlockId(blockId),
    mDeclared(false)
    {
        
    }

    virtual ~HiSymbol() {
    }
    void setDim(int dim)  {  mSymbolDim = dim; }
    
    void setBlockId(int bid)  { mBlockId = bid; }
    int getDim() const { return mSymbolDim; } 
    
    void setSymbolShape(boost::shared_ptr<HiShape> shape)
    {
        mSymbolShape = shape;
    }
    
    std::string getSymbolName() const { return mSymbolName; }
//    std::string getSymbolType() const { return mSymbolType; }
    
    bool operator==(const HiSymbol &rhs) const;
    
    void genCode(std::ostream &out) const 
    {
        genCodePrivate(out);
        mDeclared = true;
    }
    
    void hprint_w(int level) const;
    virtual void hprint(int level) const = 0 ;
    
    void pushDimensionSymbol(boost::shared_ptr<HiSymbolDimension> dim_symbol)
    {
        mDimensionSymbols.push_back(dim_symbol);
    }
    
    boost::shared_ptr<HiSymbolDimension> getDimensionSymbolAt(unsigned int index) 
    {
        assert(index < mDimensionSymbols.size());
        return mDimensionSymbols[index];
    }
    virtual HiShape getShape() const;
                             
private:
    virtual void genCodePrivate(std::ostream &out) const {}
    std::string mSymbolName;
    std::string mSymbolType;
    int mSymbolDim;
    int mBlockId;
  
    
    boost::shared_ptr<HiShape> mSymbolShape;
    std::vector<boost::shared_ptr<HiSymbolDimension> > mDimensionSymbols;
    mutable bool mDeclared; 

};


class HiSymbolConst : public HiSymbol {
public:
    virtual void hprint(int level) const ;
//    virtual HiShape getShape() const;
private:
    int mValue;
};


class HiSymbolFuncInput: public HiSymbol {
public:
    HiSymbolFuncInput(std::string str, std::string type, int dim, int blockId): 
    HiSymbol(str, type, dim, blockId)
    {}
    virtual void hprint(int level) const ;
//    virtual HiShape getShape() const;
//private:
//    std::vector<> mDimensionSymbol
};


class HiSymbolFuncOutput: public HiSymbol {
public:
    HiSymbolFuncOutput(std::string str, std::string type, int dim, int blockId): 
    HiSymbol(str, type, dim, blockId)
    {}
    virtual void hprint(int level) const ;
//    virtual HiShape getShape() const;
};




// ids that are find in part of an array index
class HiSymbolIndex: public HiSymbol {
public:
    HiSymbolIndex(std::string str, std::string type, int blockId): 
    HiSymbol(str, type, 0, blockId)
    {
        
    }
    void addDescription(boost::shared_ptr<HiSymbol> src, const ParserNode *node, int dim)
    {
        IndexDescription desc(src, node, dim);
        mDescriptions.push_back(desc);
    }
    virtual void hprint(int level) const ;
//    virtual HiShape getShape() const;
private:
    std::vector<IndexDescription> mDescriptions;
    
};


// each array generates a variable HiSymbolDimensions
class DefIndexParserNode;
class HiSymbolDimension: public HiSymbol {
public:
    HiSymbolDimension(std::string str, std::string type, int blockId, boost::shared_ptr<HiSymbol> src, int dimid, DefIndexParserNode*  expr): 
    HiSymbol(str, type, 0, blockId),
    mSource(src),
    mDimIndex(dimid),
    mExpression(expr)
    {
        
    }
    virtual void hprint(int level) const ;
    void appendShape(HiShape &shape);
private:
    boost::shared_ptr<HiSymbol> mSource;
    int mDimIndex;
    DefIndexParserNode* mExpression; // store the expression in the source code, such as I in [I]
};


class HiSymbolLocal: public HiSymbol {
public:
    HiSymbolLocal(std::string str, std::string type, int dim, int blockId): 
    HiSymbol(str, type, dim, blockId)
    {
        
    }
    virtual void hprint(int level) const ;
//    virtual HiShape getShape() const ;
};

class HiSymbolMapIter: public HiSymbol {
public:
    HiSymbolMapIter(std::string str, std::string type, int blockId): 
    HiSymbol(str, type, 0, blockId)
    {
        
    } 
    virtual void hprint(int level) const ;
//    virtual HiShape getShape() const ;
};




class HiSymbolTable {
public:
    void insertSymbol(boost::shared_ptr<HiSymbol> sym);
    bool findSymbol(const std::string &name) const ;
    boost::shared_ptr<HiSymbol> getSymbol(const std::string &name);
    
    size_t getSize() const 
    {
        return mTable.size();
    }
    
    void hprint(int level) const;
private:
    std::map<std::string, boost::shared_ptr<HiSymbol> > mTable;    
    
};



