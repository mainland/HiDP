#include "ParserNodes.h"
#include "StatementNodes.h"
#include "Utility.h"
#include <cassert>
using namespace std;
using namespace boost;


void StatementParserNode::printInOut(int level) const
{
    ParserNode::printLevel(level); 
    cout << "statement input:" ;
    for (InoutType::const_iterator it = mInputs.begin(); it != mInputs.end(); it++)
    {
        cout << (*it).first << " ";
    }
    cout << endl;
    printLevel(level); 
    cout << "statement output:" ;
    for (InoutType::const_iterator it = mOutputs.begin(); it != mOutputs.end(); it++)
    {
        cout << (*it).first << " ";
    }
    cout << endl;
    printLevel(level); 
    cout << "statement inout:" ;
    for (InoutType::const_iterator it = mInOuts.begin(); it != mInOuts.end(); it++)
    {
        cout << (*it).first << " ";
    }
    cout << endl;
    
}

void StatementParserNode::printShape(int level) const
{
    ParserNode::printLevel(level); 
    cout << "statement shape:" ;
    mShape.hprint(level);
}

MapParserNode *StatementParserNode::getParentMap()
{
    if (!mInsideMap) 
        return NULL;
    else if (!mParentMap)
    {
        BlockParserNode *pblock = mParentBlock;
        assert(pblock != 0);
        while (pblock)
        {
            if (!pblock->getMapNode())
                pblock = pblock->getParentBlock();
            else
                break;
        }
        mParentMap = pblock->getMapNode();
    }
    return mParentMap;
}


/******* DefParserNode *********/
void DefParserNode::run()
{
}

void DefParserNode::genCode(int level)
{
    // for vectors, generate var name first, then generate its dimension if necessary
    // for scalars, generate var name 
    cout << mVarType << "* " << getName() << "_" << getBelongingBlock() << endl;
}

void DefParserNode::hprint(int level)
{
    cout << "Def Node (" << getName();
    cout << ") Shape (" << mDefType;
    cout << ") Scope (" << mScope;
    cout << ") Type  (" << mVarType << ")" << endl;
    if (mIndexes)
        mIndexes->hprint_w(level+1);
//    printInOut(level);
}

DefParserNode::DefParserNode(const char *name, const char *type, bool insideMap) // scalar definition
:   StatementParserNode(StatementParserNode::Definition, insideMap),
mId(new IdParserNode(name)),
mDefType(Scalar),
mScope(Local_temp),
mVarType(type),
mIndexes(0),
mRange(0)
{
    
}

DefParserNode::DefParserNode(const char *name, IndexParserNode *indexNode, const char *type, bool insideMap) // array definition
:StatementParserNode(StatementParserNode::Definition, insideMap),
mId(new IdParserNode(name)),
mScope(Local_temp),
mVarType(type),
mIndexes(indexNode),
mRange(0)
{
    mIndexes->setId(mId);
    if (indexNode->getDim() == 0)
        mDefType = Scalar;
    else if (indexNode->getDim() == 1)
        mDefType = Array_1D;
    else if (indexNode->getDim() == 2)
        mDefType = Array_2D;
    else if (indexNode->getDim() == 3)
        mDefType = Array_3D;
    else
        assert(0); // not supported
}

DefParserNode::DefParserNode(const char *name, RangeParserNode *rangeNode, bool insideMap) // array definition
:StatementParserNode(StatementParserNode::Definition, insideMap),
mId(new IdParserNode(name)),
mScope(Map_iterator),
mIndexes(0),
mRange(rangeNode)
{
}


void DefParserNode::setScope(ScopeType scope)
{
    mScope = scope;
}

const string DefParserNode::getName() const
{
    return mId->getIdName();
}
void DefParserNode::convertToDim(Dim &dim) const
{
    assert(0); // TODO remove
 //   if (mIndexes != 0) // non scalar
  //      mIndexes->convertToDim(dim, this);
  //  else if (mRange)   // map iterators
   //     mRange->convertToDim(dim, this);
   // else
   // {
    //}
    // nothing to do if it is a scalar
}

int DefParserNode::getDimension() const
{
    if (mIndexes) 
        return mIndexes->getDim();
    else
        return 0;
}

void DefParserNode::generateSymbolType(SymbolType &type) const
{
//    type.typestr = ￼mVarType;
    type.typestr = getTypeStr();
    if (mIndexes) 
    {
//        type.dim.pushDimension(boost::shared_ptr<DimType> dim)
    }
    assert(0);  // todo from here!!!!
//    if (mIndexes)
//        mIndexes
//    type.typestr = mVarType;
//    if (mIndexes)
  //  type.dim.
}

void DefParserNode::addSymbol(HiSymbolTable *table, BlockParserNode *belongBlock) 
{
    // no need to trace upward, if the user explicitly declare a variable, it should overwrite previous one
//    shared_ptr<HiSymbol> new_symbol;
    HiSymbol *sym(0);
    if (mScope == Func_input)
        sym = new HiSymbolFuncInput(getName(), getTypeStr(), getDimension(), belongBlock->getGlobalBlockId());
    else if (mScope == Func_inout || mScope == Func_output)
        sym = new HiSymbolFuncOutput(getName(), getTypeStr(), getDimension(), belongBlock->getGlobalBlockId());
    else if (mScope == Local_temp)
        sym = new HiSymbolLocal(getName(), getTypeStr(), getDimension(), belongBlock->getGlobalBlockId());
    else if (mScope == Map_iterator)
        sym = new HiSymbolMapIter(getName(), getTypeStr(), belongBlock->getGlobalBlockId());
    else
        assert(0);
    
    shared_ptr<HiSymbol> new_symbol(sym);
    table->insertSymbol(new_symbol);
        
    // for indexes and ranges, insert them to symbol table as well
    if (mIndexes)
        mIndexes->addSymbol(table, belongBlock);
    if (mRange)
        mRange->addSymbol(table, belongBlock);
//    assert(new_symbol.use_count() > 1);
}

string DefParserNode::emitCodeForArguments(bool readOnly)
{
    return string("");// remove TODO 
/*    string code = " ";
    if (readOnly)
        code += "const ";
    if (mShape == Scalar)
        code += mVarType;
    else if (mShape == Array_1D)
        code += "HiArray<1, ";
    else if (mShape == Array_2D)
        code += "HiArray<2, ";
    else if (mShape == Array_3D)
        code += "HiArray<3, ";
    else 
        assert(0); // not supported!
    
    if (mShape != Scalar)
        code += mVarType + ">";
    
    code += " &";  // always reference
    code += getName();
    return code;*/
}



/******* AssignParserNode ********/
void AssignParserNode::run()
{
}

void AssignParserNode::dimAnalysis()
{
    if (isInsideMap())
    {
        ;
    }
    else // if outside of any mapping, either a scalar operator or a full vector operator
    {
        
    }
}

void AssignParserNode::analyzeInOut()
{
    cout << "analyzeinout in assign parser node operator" << mOp << endl;
    // only support those op for toplevel Op node (statement level)
    assert(mOp == "=" || mOp == "+=" || mOp == "-=" || mOp == "*=" || mOp == "/=" ); 
    
    // put the left hand side to output, right hand side to input, if it is +=/-=/*=//=, put the left hand side to inout
    // also consider indexes, they should be inputs
    if (mOp == "=")
    {
        assert(mLhs);
        mLhs->addIdName(mOutputs);
        mLhs->addIndexNames(mInputs);
        assert(mRhs);
        mRhs->addIdName(mInputs);
        mRhs->addIndexNames(mInputs);
    }
    else if (mOp == "+=" || mOp == "-=" || mOp == "*=" || mOp == "/=" )
    {
        // not tested yet
        assert(mLhs);
        mLhs->addIdName(mInputs);
        mLhs->addIdName(mOutputs);
        mLhs->addIndexNames(mInputs);
        assert(mRhs);
        mRhs->addIdName(mInputs);
        mRhs->addIndexNames(mInputs);
    }
}

void AssignParserNode::analyzeShape()
{
    if (mInsideMap)   // if inside map, the shape is the same as the closest map block
    {
        mShape = getParentMap()->getShape();
    }
    else // if outside a map, the shape is the same as the output
    {
//        vector<HiShapeLayer> shapes;
        cout << typeid(mLhs).name()  << " " << typeid(IdParserNode ).name() << " " << typeid(IndexParserNode).name() << endl;
        IdParserNode *id = dynamic_cast<IdParserNode *>(mLhs);
        if (id)
        {
            shared_ptr<HiSymbol> s = mParentBlock->getSymbol(id->getIdName());
            HiShape shape = s.get()->getShape();
            mShape.addLayer(shape);
            return;
        }
        IndexParserNode *index = dynamic_cast<IndexParserNode *>(mLhs);
        if (index)
        {
            HiShape shape;
            shape.appendShape(0, 1); // a unity shape
            mShape.addLayer(shape);
            return;
        }
        assert(0);
    }
}

void AssignParserNode::setStatementBelow()
{
    setStatement(this);
}

void AssignParserNode::addIdName(InoutType &inout)
{
    if (mLhs)
        mLhs->addIdName(inout);
    if (mRhs)
        mRhs->addIdName(inout);
}

void AssignParserNode::addIndexNames(InoutType &inout)
{
    if (mLhs)
        mLhs->addIndexNames(inout);
    if (mRhs)
        mRhs->addIndexNames(inout);
}


void AssignParserNode::hprint(int level)
{
    cout << "Assign Node (" << mOp;
    cout << ") inside map(" << mInsideMap;
    cout << ")" ;
    printInOut(level);
    cout << endl;
    if (mLhs)
        mLhs->hprint_w(level+1);
    if (mRhs)
        mRhs->hprint_w(level+1);
    
    printInOut(level);
    printShape(level);
}




/******* FunctionParserNode *********/
void FunctionParserNode::run()
{
}

void FunctionParserNode::setStatementBelow() 
{
    // assert(0);
}

void FunctionParserNode::analyzeInOut()
{
    if (mIsSuffix)
    {
        // only support reduce and scan
        if(mFunName=="reduce")
        {
            // args 0 : operator string node
            // args 1 : output
            // args 2 : input
            // args 3 and more : reduce range
            assert(mArgs.size() >= 4);
            mArgs[1]->addIdName(mOutputs);
            mArgs[2]->addIdName(mInputs);
        }
        else if (mFunName == "scan")
            assert(0);
        else
            assert(0);
    }
    else
    {
        assert(0); // not implemented yet
    }
}

void FunctionParserNode::analyzeShape()
{
    if (mIsSuffix)
    {
        mShape = getParentMap()->getShape();
        if(mFunName=="reduce")
        {
            // reduce(op, out, in, ranges)
            assert(mArgs.size() >= 4);
            for (int s = 3; s < mArgs.size(); s++)
            {
                DefParserNode *range_node = dynamic_cast<DefParserNode*>(mArgs[s]);
                assert(range_node);
//                RangeParserNode *range = range_node->getRange();
//                assert(range); // must be a range definition
//                string name = range_node->getName();
                MapParserNode *map_node = getParentMap();
                assert(map_node);
                mShape = map_node->getShape();
                mShape.promoteShape(range_node);
                
                
                //mArgs[s]->hprint(0);
            }
            // args after index 3 (including) are ranges
        }
        else
            assert(0);
    }
//    shared_ptr<HiShape> shape(new HiShape(getParentMap()->getShape())); // copy out 
}

void FunctionParserNode::hprint(int level)
{
    cout << "Function call Node (" << mFunName;
    cout << ") insideMap (" << mInsideMap;
    cout << ")" << endl;
    
    for (vector<ParserNode *>::iterator it = mArgs.begin(); it < mArgs.end(); it++)
    {
        (*it)->hprint_w(level+1);
    }
    printInOut(level);
    printShape(level);
    // print inout
}


FunctionParserNode::FunctionParserNode(char *name, bool insideMap):
StatementParserNode(Callfunction, insideMap),
mFunName(name),
mIsSuffix(false),
mMapBlock(NULL)
{
}

void FunctionParserNode::shiftArg(ParserNode *next)
{
    mArgs.push_back(next);
}


void FunctionParserNode::setRange(ParserNode *range)
{
    mRange = range;
}



/******* BlockParserNode *********/
void BlockParserNode::run()
{
}

void BlockParserNode::hprint(int level)
{
    cout << "Block Node" << endl; 
    printSymbolTableInfo(level);
    if (mStatements)
        mStatements->hprint_w(level+1);

}

BlockParserNode::BlockParserNode(BlockParserNode *parent, int blockId)  //
:ParserNode(),
mLevel(0),
mStatements(NULL),
mParentBlock(parent),
mGlobalBlockId(blockId),
mParentMap(NULL)
{
    if (parent)
        parent->addBlockChildren(this);
}

/*BlockParserNode::BlockParserNode(ParserNode *statements)  //
 {
 mStatements = statements;
 }*/

void BlockParserNode::setStatements(StatementParserNode *statements)
{
    mStatements = statements;
}


void BlockParserNode::analyzeInOut()
{
    cout << "analyzeinout block parser node.\n";
    StatementParserNode *stmt = mStatements;
    while(stmt)
    {
        stmt->analyzeInOut();
        stmt = dynamic_cast<StatementParserNode *>(stmt->getNext());
    }
    
    // another round to make a superset of all its statements
    stmt = mStatements;
    while(stmt)
    {
        //      stmt->analyzeInOut();
        const InoutType &in = stmt->getInputs();
        const InoutType &out = stmt->getOutputs();
        
        for (InoutType::const_iterator it = in.begin(); it != in.end(); it++)
        {
            if (mOutputs.find((*it).first) == mOutputs.end())    // if not in my output set, then it is an input
                mInputs.insert((*it));
        }
        for (InoutType::const_iterator it = out.begin(); it != out.end(); it++)
        {
            if (mInputs.find((*it).first) != mInputs.end())  // if it is already in the input list, make it an inout
            {
                mOutputs.erase((*it).first);
                mInouts.insert((*it));
            }
            else 
                mOutputs.insert((*it));
        }
        stmt = dynamic_cast<StatementParserNode *>(stmt->getNext());
        //      setInput(stmt->getInputs());
        //      setOutput(stmt->getOutputs());
        //      setInOut(stmt->getInOuts());
    }
    
}

void BlockParserNode::analyzeShape()
{
    cout << "analyzeshape block parser node.\n";
    StatementParserNode *stmt = mStatements;
    while(stmt)
    {
        stmt->analyzeShape();
        stmt = dynamic_cast<StatementParserNode *>(stmt->getNext());
    }
    


}


void BlockParserNode::appendChild(BlockParserNode *child)   //
{
    mChildren.push_back(child);
}

void BlockParserNode::addFunctionSymbol(std::map<std::string, DefParserNode*> &inout, bool readonly)
{
    assert(mParentBlock == 0); // must be the top-level
    typedef map<std::string, DefParserNode*>::iterator iter;
    for (iter it = inout.begin(); it != inout.end(); it++)
    {
        (*it).second->addSymbol(&mSymbolTable, this);
        // recursively add symbol
//        mSymTable.addSymbol((*it).second, this);
    }
  
}

void BlockParserNode::formSymbolTable()
{
    StatementParserNode *stmt = mStatements;
    while(stmt)
    {
        stmt->addSymbol(getSymbolTable(), this);
        
        stmt = dynamic_cast<StatementParserNode *>(stmt->getNext());
        
    }
    // recursively for children
    for (int i = 0; i != mChildren.size(); i++)
    {
        mChildren[i]->formSymbolTable();
    }
}

void BlockParserNode::addLocalSymbol()
{
    assert(mParentBlock == 0);
}

/*void BlockParserNode::addSymbol(DefParserNode *def)
{
    assert(0);
//    mSymTable.addSymbol(def, this);
    
//    mSymTable.add
   // type.dim  def->getDimension();
//    type.dim = 
}*/

bool BlockParserNode::findSymbol(const string &symbol, int &blockid) const
{
    blockid = -1;
    bool islocal = mSymbolTable.findSymbol(symbol);
    if (islocal)
    {
        blockid = getGlobalBlockId();
        return true;
    }
    if (symbol[0] == '_')
        islocal = mSymbolTable.findSymbol(symbol.substr(1, symbol.length()-1));
    if (islocal)
    {
        blockid = getGlobalBlockId();        
        return true;
    }
    if (mParentBlock)
        return mParentBlock->findSymbol(symbol, blockid);
    return false;
}

boost::shared_ptr<HiSymbol> BlockParserNode::getSymbol(const std::string &symbolname)
{
    shared_ptr<HiSymbol> symbol = mSymbolTable.getSymbol(symbolname);
    while (!symbol.get() && !getParentBlock())
    {
        symbol = getParentBlock()->getSymbol(symbolname);
    }
    return symbol;
}

void BlockParserNode::printSymbolTableInfo(int level) const
{
    for (int i = 0; i < level; i++)
        cout << "---";

    printf("there are %ld symbols in this block.\n", mSymbolTable.getSize());
    mSymbolTable.hprint(level);
    
//
/*    for (vector<BlockParserNode*>::const_iterator iter = mChildren.begin(); iter < mChildren.end(); iter++)
    {
        (*iter)->printSymbolTableInfo(level);
    }*/
}





/******* MapParserNode *********/
void MapParserNode::run()
{
}

void MapParserNode::setStatementBelow()
{
    setStatement(this);
    assert(mBlock);
    mBlock->setStatement(this);
}

void MapParserNode::hprint(int level)
{
    cout << "Map block Node Itervars:";
    printInOut(level);
    printShape(level);
    cout << endl;
    for (vector<DefParserNode *>::iterator iter =  mBlock->mIterVars.begin(); iter < mBlock->mIterVars.end(); iter++)
    {
        (*iter)->hprint_w(level);
    }
    mBlock->hprint_w(level+1);
    if (mSuffix)
        mSuffix->hprint_w(level+1);
    
//    printInOut(level);printShape(level);
    
}

void MapParserNode::dimAnalysis()
{
}

/*void MapParserNode::setParentMap(MapParserNode *p)
{
    mParent = p;
}*/

MapParserNode::MapParserNode(BlockParserNode *first, int level, bool insideMap):
StatementParserNode(StatementParserNode::Mapblock, insideMap),
mLevel(level),
mSuffix(NULL)
{
    mBlock = first;
    mBlock->setMapNode(this);
    mSuffix = NULL;
}

MapParserNode::MapParserNode(BlockParserNode *first, FunctionParserNode *suffix, int level, bool insideMap):
StatementParserNode(StatementParserNode::Mapblock, insideMap),
mLevel(level)
{
    mBlock = first;
    mBlock->setMapNode(this);
    mSuffix = suffix;
}

/* // trace inout
 {
 // track 
 }
 */

void MapParserNode::addSuffix(FunctionParserNode *suffix)
{
    assert(mSuffix == 0);
    mSuffix = suffix;
}

void MapParserNode::findIterators()
{
    // todo
    assert(0);
}


void MapParserNode::setStatement(StatementParserNode *node)
{
    setMyStatement(node);
    if (mSuffix)
        mSuffix->setStatement(node);
}


void MapParserNode::analyzeInOut()
{
    cout << "analyzeinout in map parser node.\n";
    // first do the block analysis
    mBlock->analyzeInOut();
    
    // need to consider suffix function
    mInputs = mBlock->getInputs();
    mOutputs = mBlock->getOutputs();
    mInOuts = mBlock->getInouts();
    if (mSuffix)
        mSuffix->analyzeInOut();
    
    const InoutType &suffixIn = mSuffix->getInputs();
    const InoutType &suffixOut = mSuffix->getOutputs();
    for (InoutType::const_iterator in = suffixIn.begin(); in != suffixIn.end(); in++)
    {
        if (mOutputs.find((*in).first) != mOutputs.end())
            continue;
        if (mInputs.find((*in).first) != mInputs.end())
            continue;
        mInputs.insert(*in);
    }
    
    for (InoutType::const_iterator out = suffixOut.begin(); out != suffixOut.end(); out++)
    {
        if (mOutputs.find((*out).first) != mOutputs.end())
            continue;
        if (mInputs.find((*out).first) != mInputs.end())
        {
            assert(0); // TODO not implemented
            continue;
        }
        mOutputs.insert(*out);
    }
    
}

void MapParserNode::analyzeShape()
{
    HiShape shape;
    if (mMapLevel > 0)  // copy parent's shape first
        mShape = getParentMap()->getShape();
//    if (mMapLevel == 0)
    {
        vector<DefParserNode *> &iters = mBlock->getMapIterators();
        assert(iters.size() > 0);
        for (vector<DefParserNode *>::iterator mapi = iters.begin(); mapi != iters.end(); mapi++)
        {
            RangeParserNode *range = (*mapi)->getRange();
            assert(range);
            vector<ParserNode *> ranges = range->getRange();
            assert(ranges.size() == 2);  // only support step 1 ranges
            shape.appendShape(mBlock, (*mapi)->getName(), shared_ptr<ParserNode>(ranges[0]), shared_ptr<ParserNode>(ranges[1]));
        }
    }
    mShape.addLayer(shape);
    
    // statement in the map
    mBlock->analyzeShape();
    
    // suffix function
    if (mSuffix)
    {
        mSuffix->analyzeShape();
        mShape = mSuffix->getShape(); // same as the suffix call
    }
    
}

void MapParserNode::addIteratorToSymbolTable()
{
    for (vector<DefParserNode *>::iterator it= mBlock->mIterVars.begin(); it < mBlock->mIterVars.end(); it++)
    {
        assert(0);
//        mBlock->addSymbol(*it);
    }
}

void MapParserNode::appendExecutionModel(HierarchyModel *model)
{
    mValidExecutionModel.push_back(model);
}

void MapParserNode::removeLastExecutionModel()
{
    mValidExecutionModel.pop_back();  
}


