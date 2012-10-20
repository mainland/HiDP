#include "DefIndexParserNode.h"
#include <iostream>
using namespace std;
using namespace boost;

/******* OpParserNode ********/
void OpParserNode::run()
{
}


#if 0
void OpParserNode::dimAnalysis()
{
/*    if (isInsideMap())
    {
        ;
    }
    else // if outside of any mapping, either a scalar operator or a full vector operator
    {
        
    }*/
}

void OpParserNode::analyzeInOut()
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


void OpParserNode::setStatementBelow()
{
    setStatement(this);
}
#endif

void OpParserNode::addIdName(InoutType &inout)
{
    if (mLhs)
        mLhs->addIdName(inout);
    if (mRhs)
        mRhs->addIdName(inout);
}

void OpParserNode::addIndexNames(InoutType &inout)
{
    if (mLhs)
        mLhs->addIndexNames(inout);
    if (mRhs)
        mRhs->addIndexNames(inout);
}

void OpParserNode::setParent(IndexParserNode *p)
{
    if (mLhs)
    {
        // TODO remove the dynamic cast, need refractoring the classes
        DefIndexParserNode *lhs = dynamic_cast<DefIndexParserNode *>(mLhs);
        lhs->setParent(p);
    }
    if (mRhs)
    {
        DefIndexParserNode *rhs = dynamic_cast<DefIndexParserNode *>(mRhs);
        rhs->setParent(p);
    }
}


void OpParserNode::addSymbol(HiSymbolTable *table, BlockParserNode *block)
{
    if (mLhs)
        mLhs->addSymbol(table, block);
    if (mRhs)
        mRhs->addSymbol(table, block);
}

void OpParserNode::emitExpression(std::stringstream &stream)
{
    assert(mLhs);
    assert(mRhs);
    stream << "(";
    mLhs->emitExpression(stream);
    stream << mOp;
    mLhs->emitExpression(stream);
    stream << ")";
}

void OpParserNode::hprint(int level)
{
    cout << "Assign Node (" << mOp;
    cout << ")" ;
//    printInOut();
    cout << endl;
    if (mLhs)
        mLhs->hprint_w(level+1);
    if (mRhs)
        mRhs->hprint_w(level+1);
}




/******* IntegerParserNode *********/
void IntegerParserNode::run()
{
}
void IntegerParserNode::hprint(int level)
{
    cout << " Integer Node (" << mVal;
    cout << ")" << endl;
}

void IntegerParserNode::emitExpression(std::stringstream &stream)
{
    stream << mVal;
}

/******* IdParserNode *********/
void IdParserNode::run()
{
}

void IdParserNode::hprint(int level)
{
    cout << "ID node (" << mId;
    cout << ")" << endl;
}

void IdParserNode::addIdName(InoutType &inout)
{
    //  inout.insert(InOut(mId));
    if (!mStatement)
        cout << mId << endl;
    else 
        cout << "non zero statement for " << mId << endl;
    assert(mStatement);
    printf("statemnt %p\n", mStatement);
    int id;
    bool inTable = mStatement->findSymbol(mId, id);
    cout << mId << " is found in symbol table? " << inTable << endl;
    inout.insert(std::make_pair<std::string, Var>(mId, Var(mId, inTable)));
}

void IdParserNode::addIndexNames(InoutType &inout)
{
    //  inout.insert(InOut(mId));
}

/*void IdParserNode::generateSymbolType(SymbolType &type)
{
    type.dim.setToScalar();
    type.typestr = string("int");
}*/

void IdParserNode::addSymbol(HiSymbolTable *table, BlockParserNode *block)
{
    int blockId;
    shared_ptr<HiSymbol> symbol;
    if (!block->findSymbol(getIdName(), blockId))
    {
        
        symbol.reset(new HiSymbolIndex(getIdName(), "int", block->getGlobalBlockId()));
        table->insertSymbol(symbol);
//        symbol.get()->
        
//        SymbolType type;
//        generateSymbolType(type);
//        symbol = new HiSymbolIndex();
//        table->insertSymbol();
    }
    else
        symbol = table->getSymbol(getIdName());
    
    // update the symbol pointer, for code generation
    mSymbol = symbol;
    
    string src_name = getParent()->getIdName();
    shared_ptr<HiSymbol> src_symbol = table->getSymbol(src_name);
    
    
    HiSymbolIndex *index_symbol = reinterpret_cast<HiSymbolIndex *>(symbol.get());
    assert(index_symbol);
    int dim = 0;
    while (getParent()->getIndex(dim) != this)
        dim++;
    
    if (index_symbol)
    {
        index_symbol->addDescription(src_symbol, getParent(), dim);
    }
    
}


void IdParserNode::emitExpression(std::stringstream &stream)
{
    stream << mId;
}

void IdParserNode::pushShape(BlockParserNode *block, std::vector<HiShapeLayer> &shapes)  //TODO const
{
    shared_ptr<HiSymbol> symbol = block->getSymbol(getIdName());
    assert(symbol != 0);
    HiShape shape = symbol.get()->getShape();
//    shapes.
//    shapes.
}
