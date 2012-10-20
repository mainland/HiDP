#include "ParserNodes.h"
#include "StatementNodes.h"
#include <cassert>
#include <typeinfo>
#include <sstream>
using namespace std;
using namespace boost;


/******* FloatParserNode *********/
void FloatParserNode::run()
{
}

void FloatParserNode::hprint(int level)
{
  cout << "Integer Node (" << mVal;
  cout << ")" << endl;
}

/******* StringParserNode *********/
void StringParserNode::run()
{
}

void StringParserNode::hprint(int level)
{
  cout << "String Node (" << mString;
  cout << ")" << endl;
}


/******* RangeParserNode *********/
void RangeParserNode::run()
{
}

void RangeParserNode::hprint(int level)
{
  cout << "Range Node" << endl;
  
  for (vector<ParserNode *>::iterator it = mRanges.begin(); it < mRanges.end(); it++)
    {
      (*it)->hprint_w(level+1);
    }
}

RangeParserNode::RangeParserNode(ParserNode *first, ParserNode *second):
  ParserNode(),
  mWildCard(false),
  mLeftClose(true),
  mRightClose(true)
{
  mRanges.push_back(first);
  mRanges.push_back(second);
}

void RangeParserNode::shift(ParserNode *third)
{
  mRanges.push_back(third);
}

void RangeParserNode::setLeftClose(bool close)
{
  mLeftClose = close;
}

void RangeParserNode::convertToDim(Dim &dim, const StatementParserNode *) const
{
    dim.setToScalar();
}

void RangeParserNode::setRightClose(bool close)
{
  mRightClose = close;
}

/******* IndexParserNode *********/
void IndexParserNode::run()
{
}

void IndexParserNode::addIdName(InoutType &inout)
{
  mId->addIdName(inout);
}

void IndexParserNode::addIndexNames(InoutType &inout) // todo const
{
    for (vector<DefIndexParserNode* >::iterator idx = mIndexes.begin(); idx < mIndexes.end(); idx++)
    {
      (*idx)->addIdName(inout);
    }
}


void IndexParserNode::setStatement(StatementParserNode *node)
{
    setMyStatement(node);
    mId->setStatement(node);
    for (std::vector<DefIndexParserNode* >::iterator it = mIndexes.begin(); it != mIndexes.end(); it++)
    {
        (*it)->setMyStatement(node);
    }
}

bool IndexParserNode::isLegalForDefinition() const
{
    return true;
/*    bool legal = true;
    for(vector<DefIndexParserNode *>::const_iterator it =  mIndexes.begin(); it != mIndexes.end(); it++)
    {
        string type = typeid(**it).name();
        IntegerParserNode in(0);
        IdParserNode id("a");
        if (type != typeid(in).name() && type != typeid(id).name())
            legal = false;
    }
    return legal;*/
}

/*void IndexParserNode::convertToDim(Dim &dim, const StatementParserNode *parent) const
{
    assert(isLegalForDefinition());
    for (vector<DefIndexParserNode *>::const_iterator idx = mIndexes.begin();idx != mIndexes.end(); idx++)
    {
        string type = typeid(*idx).name();
        if (type == "ConstParserNode") {
            IntegerParserNode *con = dynamic_cast<IntegerParserNode *>(*idx);
            dim.pushDimension(boost::shared_ptr<DimType>(new IntDim(con->getConst())));
        } else if (type == "IdParserNode")
        {
            IdParserNode *id = dynamic_cast<IdParserNode *>(*idx);
            int mapid;
            bool found = parent->findSymbol(id->getIdName(), mapid);
            assert(found);     
            dim.pushDimension(boost::shared_ptr<DimType>(new VarDim(id->getIdName(), mapid)));
        }
    }

}*/

void IndexParserNode::addSymbol(HiSymbolTable *table, BlockParserNode *belongBlock)
{
    // idname has been added to symbol
    // generate id_dim_x variable in the symbol table
    stringstream name;
    name << getIdName();
    shared_ptr<HiSymbol> src = table->getSymbol(getIdName());
    for (int i = 0; i != mIndexes.size(); i++)
    {
        stringstream name;
        name << getIdName();
        name << "_dim_";
        name << i;
        {
            int bid;
            // if hit here, replace with a while loop to get a unique name
            assert(belongBlock->findSymbol(name.str(), bid) == false);
        }
        shared_ptr<HiSymbolDimension> symbol(new HiSymbolDimension(name.str(), "int", belongBlock->getGlobalBlockId(), src, i, mIndexes[i]));
        table->insertSymbol(symbol);
        src.get()->pushDimensionSymbol(symbol);

        // also add symbol for identifier's inside the index expression
        mIndexes[i]->addSymbol(table, belongBlock);
//      mIndexes
        
    }
}
         
string IndexParserNode::getIdName() const 
{ 
    return mId->getIdName(); 
}

void IndexParserNode::hprint(int level)
{
  cout << "Index Node" << endl;
  mId->hprint_w(level+1);
  for (vector<DefIndexParserNode *>::iterator it = mIndexes.begin(); it < mIndexes.end(); it++)
    {
      (*it)->hprint_w(level+1);
    }
}


IndexParserNode::IndexParserNode(DefIndexParserNode *first):
  ParserNode(),
  mId(0)
{
  mIndexes.push_back(first);
    mIndexes[0]->setParent(this);
}

void IndexParserNode::shift(DefIndexParserNode *next)
{
  mIndexes.push_back(next);
    mIndexes[mIndexes.size()-1]->setParent(this);
}

void IndexParserNode::setId(IdParserNode *id)
{
  mId = id;
}

int IndexParserNode::getDim() const
{
  return (int)mIndexes.size();
}




/******* FuncDefParserNode *********/
FuncDefParserNode::FuncDefParserNode(char *name):
  mFuncName(name)
{
  cout << "new function name "<< mFuncName << endl;
}

void FuncDefParserNode::run()
{
}

void FuncDefParserNode::hprint(int level)
{
  cout <<"Fun Dev Node (" << mFuncName;
  cout << ")" << endl;

  /*  for (map<std::string, DefParserNode*>::iterator it =  mInputs.begin(); it < mInputs.end(); it++)
      (*it).second->hprint_w(level +1);

  for (map<std::string, DefParserNode*>::iterator it =  mOutputs.begin(); it < mOutputs.end(); it++)
      (*it).second->hprint_w(level +1);

  for (map<std::string, DefParserNode*>::iterator it =  mInouts.begin(); it < mInouts.end(); it++)
      (*it).second->hprint_w(level +1);
  */


  mFuncBlock->hprint_w(level+1);
    
    cout << "Super Statement (:" << mFusedStatement.size() << ")\n";
    for (vector<shared_ptr<SuperStatement> >::iterator it = mFusedStatement.begin(); it != mFusedStatement.end(); it++)
    {
        string shape = (*it)->getShape().emitString();
        cout <<  shape << endl;
    }

}

void FuncDefParserNode::analyzeInOut()
{
    assert(mFuncBlock);
    mFuncBlock->analyzeInOut();
}

void FuncDefParserNode::analyzeShape()
{
    assert(mFuncBlock);
    mFuncBlock->analyzeShape();
}


void FuncDefParserNode::addInput(DefParserNode *input)
{
  mInputs.insert(make_pair<string, DefParserNode*>(input->getName(), input));
}

void FuncDefParserNode::addOutput(DefParserNode *output)
{
  mOutputs.insert(make_pair<string, DefParserNode*>(output->getName(), output));
}
void FuncDefParserNode::addInout(DefParserNode *inout)
{
  mInouts.insert(make_pair<string, DefParserNode*>(inout->getName(), inout));
}

void FuncDefParserNode::setFuncBlock(BlockParserNode *block)
{
  mFuncBlock = block;
  printf("setting block %p to function.\n", block);
}

/* genereate argument list, called after inputs, outputs and inouts are added */
void FuncDefParserNode::genCodeArgument()
{
  // add symbol table
  typedef map<std::string, DefParserNode*>::iterator iter;




  mCodeArguments = "";

  for (iter it = mOutputs.begin(); it != mOutputs.end(); it++)
      mCodeArguments += (*it).second->emitCodeForArguments(false) + ",";

  for (iter it = mInouts.begin(); it != mInouts.end(); it++)
      mCodeArguments += (*it).second->emitCodeForArguments(false) + ",";

  for (iter it = mInputs.begin(); it != mInputs.end(); it++)
      mCodeArguments += (*it).second->emitCodeForArguments(true) + ",";

  cout << "function arguments is " << mCodeArguments << endl;

#ifdef _DEBUG
  mFuncBlock->printSymbolTableInfo(0);
#endif
}


/*
 shape analysis 
 for each statement in the function, analyze its shape (dim), detect conflicts
 */
void FuncDefParserNode::shapeAnalysis()
{
    //
    StatementParserNode *stmt = mFuncBlock->getStatements();
    while(stmt)
    {
        stmt->analyzeDimension();
        stmt = dynamic_cast<StatementParserNode *>(stmt->getNext());
    }
}

void FuncDefParserNode::constructSymbolTable()
{
    // first put all input/output/inout to the symbol table, add their index ID's to symbol table too (recursively)
    typedef map<std::string, DefParserNode*>::iterator iter;
    
    mFuncBlock->addFunctionSymbol(mInputs, true);
    mFuncBlock->addFunctionSymbol(mOutputs, false);
    mFuncBlock->addFunctionSymbol(mInouts, false);
  
    mFuncBlock->formSymbolTable(); // recursively form symbol table
}

//void FuncDefParserNode::analyzeInOut()
//{
    
//}

size_t FuncDefParserNode::getNumSuperStatement() const
{
    return mFusedStatement.size();
}

void FuncDefParserNode::fuseToSuperStatement(int optlevel)
{
  // if optimization level is 0, one block is mapped into one superstatement
  mFusedStatement.push_back(shared_ptr<SuperStatement>(new SuperStatement()));

  shared_ptr<SuperStatement>curSuper = mFusedStatement[mFusedStatement.size()-1];
  StatementParserNode *stmt = mFuncBlock->getStatements();      
/*  if (optlevel == 0)
    {
      // make every statement a superstatement
      assert(0); // todo 
      while (stmt)
        {
          mFusedStatement[0]->addStatement(stmt);
          stmt = dynamic_cast<StatementParserNode *>(stmt->getNext());
        }
      printf("there are %d statement in the super statement.\n", mFusedStatement[0]->getNumStatements());
      return;
    }*/

  // for other optlevel, try fuse
  // first analyze input/output for every statement below
  stmt = mFuncBlock->getStatements();      

 /* while (stmt)
    {
      //      printf("here 2.\n");
      stmt->analyzeInOut();
      stmt = dynamic_cast<StatementParserNode *>(stmt->getNext());
    }

  stmt = mFuncBlock->getStatements();*/
    
  curSuper->addStatement(stmt);
  stmt = dynamic_cast<StatementParserNode *>(stmt->getNext());
  while (stmt)
    {
      if (curSuper->compatible(stmt))
        {
          curSuper->addStatement(stmt);
          cout << "fuse two statements!!!!" << endl;
        }
      else 
        {
          mFusedStatement.push_back(shared_ptr<SuperStatement>(new SuperStatement()));
          curSuper = mFusedStatement[mFusedStatement.size()-1];
          curSuper->addStatement(stmt);
        }
      stmt = dynamic_cast<StatementParserNode *>(stmt->getNext());
    }
}


bool FuncDefParserNode::mapExecution(boost::shared_ptr<class HierarchyMode> mode)
{
    bool success = true;
    for (vector<shared_ptr<SuperStatement> >::iterator ss = mFusedStatement.begin(); ss != mFusedStatement.end(); ss++)
    {
        if (!(*ss)->mapExecution(mode))
            return false;
    }
    return true;
}


