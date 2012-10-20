#pragma once

#include "HiSymbol.h"
#include "SuperStatementNodes.h"
#include "Parser.h"
#include "SymbolTable.h"
#include "HiSymbol.h"
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib>
#include "DefIndexParserNode.h"
#include <boost/shared_ptr.hpp>



//typedef std::map<std::string, boost::shared_ptr<HiSymbol> > HiSymbolTable;




class FloatParserNode : public ParserNode {
 public:
 FloatParserNode(double val):
  ParserNode(),
  mVal(val)
  {
    //    printf("init integer node %d.\n", mVal);
  }

  virtual void run();
  // hierarchically print
  virtual void hprint(int level);

 private:
  double mVal;
};

class StringParserNode : public ParserNode {
 public:
 StringParserNode(const char *c):
  ParserNode(),
  mString(c)
  {
  }

  virtual void run();
  // hierarchically print
  virtual void hprint(int level);

 private:
  std::string mString;
};



class RangeParserNode : public ParserNode {
 public:
 RangeParserNode():  // wildcard
  ParserNode(),
  mWildCard(true),
    mLeftClose(true),
    mRightClose(false)
  {
  }
  RangeParserNode(ParserNode *first, ParserNode *second);

  void shift(ParserNode *third);
  void setLeftClose(bool close);
  void setRightClose(bool close);
  
  void convertToDim(Dim &dim, const StatementParserNode*) const;
    
  virtual void run();
  // hierarchically print
  virtual void hprint(int level);

  virtual void setStatement(StatementParserNode *node)
  {
    setMyStatement(node);
    for (std::vector<ParserNode *>::iterator it = mRanges.begin(); it != mRanges.end(); it++)
      {
        (*it)->setStatement(node);
      }
  }
    std::vector<ParserNode *> &getRange() { return mRanges; }
 private:
  bool mWildCard;
  bool mLeftClose;
  bool mRightClose;

  std::vector<ParserNode *> mRanges;
};

class StatementParserNode;
class DefIndexParserNode;
class IdParserNode;
class IndexParserNode : public ParserNode {
 public:
  explicit IndexParserNode(DefIndexParserNode *);

  void shift(DefIndexParserNode *third);
  void setId(IdParserNode *id);
  int getDim() const;
    
    std::string getIdName() const ;// { return mId->getIdName(); }

  virtual void run();
  // hierarchically print
  virtual void hprint(int level);

  virtual void addIdName(InoutType &inout);
  virtual void addIndexNames(InoutType &inout);

    virtual void setStatement(StatementParserNode *node);    
    // all indexes must be id or const
    bool isLegalForDefinition() const;

//    void convertToDim(Dim &dim, const StatementParserNode *parent) const;
    void addSymbol(HiSymbolTable *table, BlockParserNode *belongBlock);
    DefIndexParserNode *getIndex(unsigned int id)  const { assert(id < mIndexes.size()); 
        return mIndexes[id]; }
 private:
  IdParserNode *mId;
//protected:
    std::vector<DefIndexParserNode*> mIndexes;
};



class BlockParserNode: public ParserNode {
 public:
  BlockParserNode(BlockParserNode *parent, int blockId);

  virtual void run();
  // hierarchically print
  virtual void hprint(int level);

  bool findSymbol(const std::string &symbol, int &blockId) const;
    boost::shared_ptr<HiSymbol> getSymbol(const std::string &symbol);
//  void addSymbol(DefParserNode *def);
    void addStatement(ParserNode *stmts);
  void appendChild(BlockParserNode *);

  // FOR DEBUG
  void printSymbolTableInfo(int level) const;
  void setStatements(StatementParserNode *statements);
  StatementParserNode *getStatements() { return mStatements; }

  void addIterVars(DefParserNode *node) { mIterVars.push_back(node);  }
  void analyzeInOut();
    void analyzeShape();
    
    void addFunctionSymbol(std::map<std::string, DefParserNode*> &inout, bool readonly);
    void addLocalSymbol();
  
  const InoutType &getInputs() const { return mInputs; }
  const InoutType &getOutputs() const { return mOutputs; }
  const InoutType &getInouts() const { return mInouts; }

  InoutType &getInputs()  { return mInputs; }
  InoutType &getOutputs()  { return mOutputs; }
  InoutType &getInouts()  { return mInouts; }

    void addBlockChildren(BlockParserNode *block) { mChildren.push_back(block); }
  // for backend
  friend class MapParserNode;
    int getGlobalBlockId() const { return mGlobalBlockId; }
    
    HiSymbolTable *getSymbolTable() { return &mSymbolTable; }
    void formSymbolTable();
//    HiSymbolTable *getSymbolTable()   { return &mSymbolTable; }
//    std::string getGlobalBlockIdStr() const { return std::string(itoa(mGlobalBlockId)); }
    void setMapNode(struct MapParserNode *m)   { mParentMap = m; }
    struct MapParserNode *getMapNode() { return mParentMap; }
    struct BlockParserNode *getParentBlock() const{ return mParentBlock; }
    std::vector<DefParserNode *> &getMapIterators() { return mIterVars; }
 private:
  //  ParserNode *mNext;  // TODO support multiple next?
  int mLevel;
  StatementParserNode *mStatements;
//  SymbolTable mSymTable;
  HiSymbolTable mSymbolTable;
    struct MapParserNode *mParentMap;
//    HiSymbolTable* mSymbolTable;
  struct BlockParserNode *mParentBlock;   // parent block
  std::vector<DefParserNode *> mIterVars;
  std::vector<BlockParserNode*> mChildren;  // vector of block children

  InoutType mInputs;
  InoutType mOutputs;
  InoutType mInouts;
    
    int mGlobalBlockId; // a unique block id, used for symbol table identification
};


                                        
class FunDefParserNode : public ParserNode {
 public:
  explicit FunDefParserNode(const char *);

  virtual void run();
  // hierarchically print
  virtual void hprint(int level);

 private:
  std::string mName;
  std::vector<DefParserNode*> mInOuts;
  BlockParserNode *mFuncBlock;
  FunDefParserNode *mNext;
};


class BlockParserNode;
class FuncDefParserNode : public ParserNode {
 public:
  explicit FuncDefParserNode(char *);

  virtual void run();
  // hierarchically print
  virtual void hprint(int level);


  void addInput(DefParserNode *);
  void addOutput(DefParserNode *);
  void addInout(DefParserNode *);

  void setFuncBlock(BlockParserNode *);
  void genCodeArgument();
  void fuseToSuperStatement(int optlevel);
  void shapeAnalysis();
  const std::string getFuncName() const { return mFuncName; }

    // middle end
    void constructSymbolTable();
//    void analyzeInOut();

  // for backend
  virtual void analyzeInOut();
    void analyzeShape();

    size_t getNumSuperStatement() const;

    bool mapExecution(boost::shared_ptr<class HierarchyMode> mode) ;
 private:
  std::string mFuncName;
  BlockParserNode *mFuncBlock;
  std::map<std::string, DefParserNode*> mInputs;
  std::map<std::string, DefParserNode*> mOutputs;
  std::map<std::string, DefParserNode*> mInouts;

  std::string mCodeArguments;

  std::vector<boost::shared_ptr<SuperStatement> > mFusedStatement;
};
