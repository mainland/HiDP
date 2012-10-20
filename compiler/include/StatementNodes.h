#pragma once

#include "Parser.h"
//#include "SymbolTable.h"

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include "StatementShape.h"
#include "HiSymbol.h"
#include "HiShape.h"
#include <boost/shared_ptr.hpp>


class DefParserNode;
class DefIndexParserNode;

class AssignParserNode : public StatementParserNode {
public:
    AssignParserNode(ParserNode *lhs, ParserNode *rhs, const char *op, bool insideMap):
    StatementParserNode(StatementParserNode::Assignment, insideMap),
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
    
    virtual void dimAnalysis();
    
    void setLhsFirst(bool firstShown) 
    { 
        mLhsFirstShow = firstShown; 
    }
    
    virtual void analyzeInOut();
    virtual void analyzeShape();
    
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
    
    virtual void setStatementBelow();
    
    virtual void addIdName(InoutType &inout);
    virtual void addIndexNames(InoutType &inout);
    
private:
    ParserNode *mLhs;
    ParserNode *mRhs;
    std::string mOp;
    bool mLhsFirstShow;
};


class HierarchyModel;
class BlockParserNode;
class FunctionParserNode;

class MapParserNode : public StatementParserNode {
public:
    MapParserNode(BlockParserNode *block, int level, bool insideMap);
    MapParserNode(BlockParserNode *block, FunctionParserNode *suffix, int level, bool insideMap);
    
    virtual void run();
    // hierarchically print
    virtual void hprint(int level);
    
    void addSuffix(FunctionParserNode *suffix);
    void findIterators(); 
    
    // for the execution model mapping
    void appendExecutionModel(HierarchyModel *);
    void removeLastExecutionModel();
    void addIteratorToSymbolTable();
    
    virtual void setStatementBelow();
    virtual void analyzeInOut();
    virtual void analyzeShape();
    virtual void dimAnalysis();
    virtual void setStatement(StatementParserNode *node);    
    
//    void setParentMap(MapParserNode *p);
//    MapParserNode *getParentMap();
private:
    BlockParserNode *mBlock;
    FunctionParserNode *mSuffix;
    
    
//    MapParserNode *mParent; // level
    int mLevel;
    //  std::set<std::string> mFunctionsInside;
    std::vector<HierarchyModel*> mValidExecutionModel;
    
    
    
};

class MapParserNode;
class FunctionParserNode : public StatementParserNode {
public:
    FunctionParserNode(char *, bool insideMap);
    
    void shiftArg(ParserNode *third);
    void setRange(ParserNode *range);
    
    void setSuffix() { 
        mIsSuffix = true; 
    }
    
    virtual void run();
    // hierarchically print
    virtual void hprint(int level);
    
    virtual void analyzeInOut();
    virtual void analyzeShape();
    virtual void setStatementBelow();
    
    virtual void setStatement(StatementParserNode *node)
    {
        setMyStatement(node);
        std::cout << "arg size " <<  mArgs.size() << std::endl;
        for (std::vector<ParserNode *>::iterator it = mArgs.begin(); it != mArgs.end(); it++)
        {
            assert(*it);
            (*it)->hprint_w(0);
            (*it)->setStatement(node);
        }
        if (mRange)
            mRange->setStatement(node);
    }
    
    void setParentMap(MapParserNode *map)
    {
        mMapBlock = map;
    }
    
    virtual struct MapParserNode *getParentMap()
    {
        return mMapBlock;
    }
   // void setMapNode(MapParserNode *map)
//    {
//        mMapBlock = map;
//    }
private:
    std::string mFunName;
    std::vector<ParserNode *> mArgs;
    ParserNode *mRange;
    bool mIsSuffix;
    MapParserNode *mMapBlock;
};


class SymbolTable;
class BlockParserNode;
class DefParserNode : public StatementParserNode {
public:
    enum DefType{
        Scalar,
        Array_1D,
        Array_2D,
        Array_3D
    };
     enum ScopeType{
        Map_iterator,
        Local_temp,
        Func_input,
        Func_output,
        Func_inout
    } ;
    
    DefParserNode(const char *name, const char *type, bool insideMap); // scalar definition
    DefParserNode(const char *name, IndexParserNode *indexNode, const char *type, bool insideMap); // array definition
    DefParserNode(const char *name, RangeParserNode *indexNode, bool insideMap); // map iterator definition
    
    virtual void run();
    
    virtual void genCode(int level);
    // hierarchically print
    virtual void hprint(int level);
    
    void setType(const char *);
    void setScope(ScopeType scope);
    ScopeType getScope() const { return mScope; }
    
    const std::string getName() const;
    
    void convertToDim(Dim &Dim) const;
    DefType getDefType() const  { return mDefType; } 

    std::string getTypeStr() const { return mVarType; }
    //for code generation
    std::string emitCodeForArguments(bool readOnly); 
    
    // 
    int getDimension() const;
    virtual void setStatementBelow() { }


    
//    void addToSymbolTable(SymbolTable *table, BlockParserNode *belongBlock);
    void generateSymbolType(SymbolType &type) const;
    virtual void addSymbol(HiSymbolTable *table, BlockParserNode *belongBlock);
    RangeParserNode *getRange() { return mRange; }
private:
    //std::string mName;
    IdParserNode *mId;
    DefType mDefType;
    ScopeType mScope;
    std::string mVarType;
    IndexParserNode *mIndexes;
    RangeParserNode *mRange;
};


