#pragma once

//#include <boost/shared_ptr.hpp>
#include "Dim.h"
#include "StatementShape.h"
#include <vector>
#include <set>
#include <map>
#include <string>
#include <cassert>
#include "HiSymbol.h"
#include "HiShape.h"

typedef struct YYLTYPE
{
    int first_line;
    int first_column;
    int last_line;
    int last_column;
} YYLTYPE;

class InOut{
 public:
 explicit InOut(std::string &name):mName(name)
    {
      mScalar = (mName[0] == '_');
    }
  InOut(const InOut &rhs)
    {
      mName = rhs.mName;
      mScalar = rhs.mScalar;
    }

  InOut &operator=(InOut &rhs)
    {
      mName = rhs.mName;
      mScalar = rhs.mScalar;
      return *this;
    }

  std::string mName;

  bool mScalar;
  bool operator<(InOut &rhs) const
    {
      return mName < rhs.mName;
    }
};

class Var{
 public:
 Var(std::string st, bool inTable):mName(st),
    mInSymbolTable(inTable)
    {
      mScalar =  (st[0] == '_');
      mInsideMap = false;  // todo
    }

  std::string mName;
  bool mScalar;
  bool mInsideMap;
  bool mInSymbolTable;
    
};

struct InOutCompare
{
  template <class T>
  bool operator()(const T& pX, const T& pY) const
  {
    return pX.mName < pY.mName;
  }
};

//typedef std::set<InOut, InOutCompare> InoutType;
typedef std::map<std::string,Var> InoutType;
//typedef std::make_pair<std::string, Var> 

class StatementParserNode;
class SymbolTable;
class BlockParserNode;

class ParserNode {
 public:
 ParserNode(): mNext(0), mMapLevel(0), mStatement(0) {}

    virtual ~ParserNode() {}
  virtual void run() = 0;

  void setNext(ParserNode *);
  void setMapLevel(int level) { mMapLevel = level; }

  // hierarchically print
  void hprint_w(int level);

  virtual void hprint(int level) = 0;
  // get the identification number, only applies to IdNode

  // non-top level 
  virtual void addIdName(InoutType &inout) {}
  // get the identification number for indexes, only applies to Index node
  virtual void addIndexNames(InoutType &inout) { }


  // used to identify shape 
  virtual void emitExpression(std::stringstream &stream) {} 

  void printLevel(int level) const; 
  void printNext(int level); 

  ParserNode *getNext()  { return mNext; }

  virtual void setStatement(StatementParserNode *node) { setMyStatement(node); } 
  void setMyStatement(StatementParserNode *node) { mStatement = node; }

    // symbol table
    virtual void addSymbol(HiSymbolTable *table, BlockParserNode *block) 
    {
            // do nothing by default
    } 
    
    // used for shape analysis (recursive)
    virtual void pushShape(struct BlockParserNode *block, std::vector<HiShapeLayer> &shapes) {} // do nothing by default
 private:

  class ParserNode *mNext;

 protected:
  int mMapLevel;
  struct StatementParserNode *mStatement; // the pointer to the statement node that contains this node  


};


class BlockParserNode;
class StatementParserNode :public ParserNode{
 public:
  enum StatementType{
    Assignment,
    Definition,
    Mapblock,
    Callfunction,
    Super
  };

  
 StatementParserNode(StatementType type, bool insideMap, YYLTYPE *srcLoc = 0):
  ParserNode(),
    mType(type),
    mInsideMap(insideMap),
    mParentBlock(0),
    mParentMap(0),
    mShape(),

    mSuper(NULL)
    { 
//    mInputs.clear();
//    mOutputs.clear();
//    mInOuts.clear();
        
    // store location of the source code
    if (srcLoc != 0)
    {
        mDebugInfo = *srcLoc;
    }
  }

  virtual void run() = 0;

  virtual void hprint(int level) = 0 ;

  virtual void analyzeDimension() {}   // TODO make it pure 

  StatementType getType() const { return mType; }

  void genInput();  
  void genOutput();
  virtual void analyzeInOut() {} // do nothing by default
    virtual void analyzeShape() {} // do nothing by default

    HiShapeLayer getShape() { return mShape; }  // copy it out
    
  // for debug
 // void printInOut() const;

  const InoutType &getInputs() const { return mInputs; }
  const InoutType &getOutputs() const { return mOutputs; }
  const InoutType &getInOuts() const { return mInOuts; }

    void setSuperStatement(class SuperStatement *supers) { mSuper = supers; }
  virtual void setStatementBelow() = 0;
  bool findSymbol(std::string &symbol, int &mapid) const ;
  void setParentBlock(BlockParserNode *parentb)  { mParentBlock = parentb; }

  bool isInsideMap() const  {  return mInsideMap; }
    
    void DimAnalysis()  {} // do nothing by default
    
    const  BlockParserNode *  getBelongingBlock() const { return mParentBlock;}
    BlockParserNode *getBelongingBlock() { return mParentBlock;}
    
    void printInOut(int level) const ;
    void printShape(int level) const;
    
    virtual struct MapParserNode *getParentMap();
 private:
  StatementType mType;

  Dim mDim;
 protected:
  bool mInsideMap;
  InoutType mInputs;
  InoutType mOutputs;
  InoutType mInOuts;

  Dim mDimension; // dimension of this 
  struct BlockParserNode *mParentBlock;   // parent block
    struct MapParserNode *mParentMap;


//  StatementShape mStmtShape;
    
    HiShapeLayer mShape;
    
    YYLTYPE mDebugInfo;
    class SuperStatement *mSuper;
};


