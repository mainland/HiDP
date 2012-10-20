#include "HiSymbol.h"
#include "HiShape.h"
#include "ParserNodes.h"
using namespace std;
using namespace boost;

void HiSymbol::hprint_w(int level) const
{
    printLevel(level);
    hprint(level);
    std::cout << mSymbolName << " " << mSymbolType << " dim(" << mSymbolDim << ") blockId(" << mBlockId << ")" ;
    assert(mSymbolDim == mDimensionSymbols.size());
    for (int i = 0; i != mSymbolDim; i++)
    std::cout << "( " << mDimensionSymbols[i].get()->getSymbolName() << ") ";
    std::cout << std::endl ;
    
}

    
bool HiSymbol::operator==(const HiSymbol &rhs) const
{
    return ((mBlockId == rhs.mBlockId) && (mSymbolName == rhs.mSymbolName)
            && (mSymbolType == rhs.mSymbolType) && (mSymbolDim == rhs.mSymbolDim));
}

HiShape HiSymbol::getShape() const
{
    HiShape shape;
    if (getDim() == 0)
    {
        shape.appendShape(0, 1);
        return shape;
    }
    assert(mDimensionSymbols.size() == getDim());
    for (int d = 0; d != getDim(); d++)
    {
        mDimensionSymbols[d].get()->appendShape(shape);
    }
    return shape;
}


void HiSymbolTable::insertSymbol(boost::shared_ptr<HiSymbol> sym)
{
    cout << "inserting " << sym.get()->getSymbolName() << endl;
    mTable.insert(std::pair<std::string, boost::shared_ptr<HiSymbol> > (sym.get()->getSymbolName(), sym));
}


bool HiSymbolTable::findSymbol(const std::string &name) const 
{
    if (mTable.find(name) != mTable.end())
        return true;
    return false;
    
}

shared_ptr<HiSymbol> HiSymbolTable::getSymbol(const std::string &name)
{
    shared_ptr<HiSymbol> symbol;
    if (mTable.find(name) != mTable.end())
    {
        symbol = mTable[name];
    }
    return symbol;
}

void HiSymbolTable::hprint(int level) const
{
    for (int i = 0; i < level; i++)
        cout << "---";
    cout << "\n";
    for (map<string, boost::shared_ptr<HiSymbol> >::const_iterator it = mTable.begin(); it != mTable.end(); it++)
    {
//        cout << (*it).first << " "; // TODO
        (*it).second.get()->hprint_w(level);
    }
    cout << endl;
    /*for (map<string, DefParserNode *>::const_iterator it = mTable.begin(); it != mTable.end(); it++)
     {
     cout << (*it).first << " ";
     }*/
    cout << "\n";

}


void HiSymbolConst::hprint(int level) const
{
    cout << "HiSymbolConst " ;
}



void HiSymbolIndex::hprint(int level) const
{
    cout << "HiSymbolIndex " ;
    for (int i = 0; i != mDescriptions.size(); i++)
        cout << "(" << mDescriptions[i].mSource.get()->getSymbolName() << " dim(" <<  mDescriptions[i].mDimIndex << ") ) ";
}


void HiSymbolFuncInput::hprint(int level) const
{
    cout << "HiSymbolFuncInput " ;
}


void HiSymbolFuncOutput::hprint(int level) const
{
    cout << "HiSymbolFuncOutput " ;
}

void HiSymbolLocal::hprint(int level) const
{
    cout << "HiSymbolLocal " ;    
}

void HiSymbolDimension::hprint(int level) const
{
    cout << "HiSymbolDimension " ;    
}

void HiSymbolDimension::appendShape(HiShape &shape) 
{
    shared_ptr<ParserNode> node;
//    ParserNode *tmp = dynamic_cast<ParserNode *>(mExpression);
    node.reset(mExpression); //reinterpret_cast<ParserNode *>(mExpression));
    shape.appendShape(0, mSource, node);
    //shape.appendShape(0, shared_ptr<ParserNode>(reinterpret_cast<ParserNode*>(mExpression)));
}

void HiSymbolMapIter::hprint(int level) const
{
    cout << "HiSymbolMapIter " ;    
}


