
#pragma once

//#include <boost/shared_ptr.hpp>
#include "Dim.h"
#include "StatementShape.h"
#include <vector>
#include <set>
#include <map>
#include <string>
#include <sstream>
#include <cassert>
#include "ParserNodes.h"
#include "SymbolTable.h"

class SymbolTable;
class BlockParserNode;
class IndexParserNode;

class DefIndexParserNode :public ParserNode{
public:
    explicit DefIndexParserNode(IndexParserNode *parent):
    ParserNode(),
    mParent(parent)
    {
        
    }
    
    virtual ~DefIndexParserNode() {}
    virtual void run() = 0;
    
    virtual void hprint(int level) = 0 ;

    
    virtual void addSymbol(HiSymbolTable *table, BlockParserNode *block) = 0;
    virtual void setParent(IndexParserNode *p) { mParent = p; }

    const IndexParserNode *getParent() const { return mParent;  }
    void setDimIndex(int dimId)  { mDimIndex = dimId; }
    
    virtual void emitExpression(std::stringstream &stream) = 0;
private:

protected: 
    IndexParserNode *mParent;
    int mDimIndex;
};


class OpParserNode : public DefIndexParserNode {
public:
    OpParserNode(ParserNode *lhs, ParserNode *rhs, const char *op, bool insideMap):
    DefIndexParserNode(0),
    mLhs(lhs),
    mRhs(rhs),
    mOp(op),
    mLhsFirstShow(false)
    {
        //    printf("init integer node %d.\n", mVal);
    }
    
    virtual void run();
    // hierarchically print
    virtual void hprint(int level);
    
   // virtual void dimAnalysis();
    
    void setLhsFirst(bool firstShown) 
    { 
        mLhsFirstShow = firstShown; 
    }
    
//    virtual void analyzeInOut();
    ParserNode *getLeft() { return mLhs; }
    ParserNode *getRight() { return mRhs; }
    
    virtual void setStatement(StatementParserNode *node)
    {
        setMyStatement(node);
        if (mLhs)
        {
            std::cout << "set statement for left." << std::endl;
            mLhs->hprint_w(0);
            mLhs->setStatement(node);
        }
        if (mRhs)
        {
            std::cout << "set statement for right." << std::endl;
            mRhs->hprint_w(0);
            mRhs->setStatement(node);
        }
    }
    
   // virtual void setStatementBelow();
    
    virtual void addIdName(InoutType &inout);
    virtual void addIndexNames(InoutType &inout);
    
    virtual void addSymbol(HiSymbolTable *table, BlockParserNode *block);
    virtual void setParent(IndexParserNode *p);
    
    virtual void emitExpression(std::stringstream &stream);
private:
    ParserNode *mLhs;
    ParserNode *mRhs;
    std::string mOp;
    bool mLhsFirstShow;
};


class IntegerParserNode : public DefIndexParserNode {
public:
    IntegerParserNode(int val):
    DefIndexParserNode(0),
    mVal(val)
    {
        //    printf("init integer node %d.\n", mVal);
    }
    
    virtual void run();
    
    // hierarchically print
    virtual void hprint(int level);
    
    int getConst() const { return mVal; }
    virtual void addSymbol(HiSymbolTable *table, BlockParserNode *block)
    {
        // do nothing
    }
    virtual void emitExpression(std::stringstream &stream);
private:
    int mVal;
};

class mSymbol;
class IdParserNode : public DefIndexParserNode {
public:
    IdParserNode(const char *id):
    DefIndexParserNode(0),
    mId(id)
    {
        //    printf("init integer node %d.\n", mVal);
    }
    
    virtual void run();
    // hierarchically print
    virtual void hprint(int level);
    
    virtual void addIdName(InoutType &inout);
    virtual void addIndexNames(InoutType &inout);
    
    virtual std::string &getIdName() 
    { 
        return mId;
    }
    
//    void generateSymbolType(SymbolType &type);
    virtual void addSymbol(HiSymbolTable *table, BlockParserNode *block);
    
    virtual void emitExpression(std::stringstream &stream);
    
    virtual void pushShape(BlockParserNode *block, std::vector<HiShapeLayer> &shapes);
private:
    std::string mId;
    boost::shared_ptr<HiSymbol> mSymbol;
};


