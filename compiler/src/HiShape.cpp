
#include "HiShape.h"
#include "DefIndexParserNode.h"
#include "ParserNodes.h"
#include "StatementNodes.h"
using namespace boost;
using namespace std;

void HiShape::appendShape(int left, int right)
{
//    mLeftLimiters.push_back(shared_ptr<ParserNode>(new IntegerParserNode(left)));
//    mRightLimiters.push_back(shared_ptr<ParserNode>(new IntegerParserNode(right)));

    HiShapeItem item;
    item.mLeftLimiter = shared_ptr<ParserNode>(new IntegerParserNode(left));
//    item.mSymbol.reset(); // no symbol
     item.mRightLimiter = shared_ptr<ParserNode>(new IntegerParserNode(right));
    mShape.push_back(item);
}

void HiShape::appendShape(int left, shared_ptr<HiSymbol> symbol, shared_ptr<ParserNode> right)
{
//    mLeftLimiters.push_back(shared_ptr<ParserNode >(new IntegerParserNode(left)));
//    mRightLimiters.push_back(right);
    
    HiShapeItem item;
    item.mLeftLimiter = shared_ptr<ParserNode>(new IntegerParserNode(left));
    item.mRightLimiter = right;
    item.mSymbol = symbol;
    mShape.push_back(item);
}   

void HiShape::appendShape(BlockParserNode *block, const string &id, shared_ptr<ParserNode> left, shared_ptr<ParserNode> right)
{
//    mLeftLimiters.push_back(left);
//    mRightLimiters.push_back(right);
    
    HiShapeItem item;
    item.mLeftLimiter = left;
    item.mSymbol = block->getSymbol(id);
    assert(item.mSymbol.use_count() > 1);
    item.mRightLimiter = right;
    mShape.push_back(item);
}

void HiShape::appendShape(const HiShapeItem &item)
{
    mShape.push_back(item);
}


void HiShape::removeShape(list<HiShapeItem>::iterator iter)
{
    cout << "symbol " << (*iter).mSymbol.get()->getSymbolName() << endl;
    cout << "user count " << (*iter).mSymbol.use_count() << endl;
    assert((*iter).mSymbol.use_count() > 1);
    mShape.erase(iter);
    if (mShape.size() == 0)
    {
        appendShape(0, 1); // if empty, add a unity shape
    }
}

void HiShape::emitToString(std::stringstream &ss) const
{
    for (list<HiShapeItem>::const_iterator l = mShape.begin(); l != mShape.end(); l++)
    {
        ss << "(";
        (*l).mLeftLimiter.get()->emitExpression(ss);
        ss << ",";
        (*l).mRightLimiter.get()->emitExpression(ss);
        ss << ")";
    }
}

bool HiShape::isCompatible(boost::shared_ptr<class HierarchyMode> model) const
{
    // TODO check suitable parallel range and primitive capability
    return true;
}

list<HiShapeItem>::iterator HiShape::findShape(const std::string &name, bool &found)
{
    list<HiShapeItem>::iterator it = mShape.begin();
    found = true;
    while(it != mShape.end())
    {
        if ((*it).mSymbol.get()->getSymbolName() == name)
            return it;
        else
            it++;
    }
    found = false;
    return it;
}

void HiShape::print() const
{
    stringstream stream;
    for (list<HiShapeItem>::const_iterator it = mShape.begin(); it != mShape.end(); it++)
    {
        (*it).mLeftLimiter.get()->emitExpression(stream);
        stream << " ";
        (*it).mRightLimiter.get()->emitExpression(stream);
        stream << " ";
        stream << "symbol(" << (*it).mSymbol.get()->getSymbolName() << "), ";
    }
    cout << stream.str();
    cout << "\n";
}

bool HiShape::operator==(const HiShape &rhs)
{
    stringstream ll; 
    stringstream lr;
    stringstream rl;
    stringstream rr;
    if (getSize() != rhs.getSize()) return false;
    list<HiShapeItem>::const_iterator r = rhs.mShape.begin();
    for (list<HiShapeItem>::const_iterator l = mShape.begin(); l != mShape.end(); l++, r++)
    {
        ll.clear();  lr.clear();  rl.clear();  rr.clear();
        
        (*l).mLeftLimiter.get()->emitExpression(ll);
        (*l).mRightLimiter.get()->emitExpression(lr);
        (*r).mLeftLimiter.get()->emitExpression(rl);
        (*r).mRightLimiter.get()->emitExpression(rr);
        if ((ll.str() != rl.str()) || (rl.str() != rr.str()))
            return false;
    }
    return true;
}

void HiShapeLayer::hprint(int level) const
{
    printLevel(level);
    cout << "{ " ;
    for (int i = 0; i != mLayers.size(); i++)
    {

        cout << "[";
       // for (int j = 0; j != mLayers[i].get()->getSize(); j++)
//        {
            mLayers[i].print();
  //      }
        cout << "]";

    }
    cout << "}";
}




bool HiShapeLayer::isCompatible(const HiShapeLayer &rhs)
{
    assert(0);
    return true;
}

void HiShapeLayer::addShapeAtIndex(const HiShapeItem &item, int index)
{
    if (index == mLayers.size())
    {
        HiShape shape;
        shape.appendShape(item);
        addLayer(shape);
        
    } else if (index == (mLayers.size()-1))
    {
        mLayers[index-1].appendShape(item);
        
    } else {
        assert(0);
    }
        
}

string HiShapeLayer::emitString() const
{
    stringstream ss;
    ss << "{";
    for (vector<HiShape>::const_iterator it = mLayers.begin(); it != mLayers.end(); it++)
    {
        ss << "[";
        (*it).emitToString(ss);
        ss << "]";
    }
    ss << "}";
    return ss.str();
}

string HiShapeLayer::emitFlatString() const
{
    stringstream ss;
    for (vector<HiShape>::const_iterator it = mLayers.begin(); it != mLayers.end(); it++)
    {
        (*it).emitToString(ss);
    }
    return ss.str();
}


void HiShapeLayer::fuseShape(const HiShapeLayer &shape)
{
//    assert(getNumLayers() <= shape.getNumLayers());
    if (getNumLayers() < shape.getNumLayers())
        *this = shape;
//    else
//        assert(*this == shape);
}

bool HiShapeLayer::operator==(const HiShapeLayer &rhs) const
{
    string l = emitString();
    string r = rhs.emitString();
    return (l == r);
}

void HiShapeLayer::promoteShape(struct DefParserNode *range)
{
    assert(range);
    string name = range->getName();
    int index = 0;
    for (vector<HiShape>::iterator it = mLayers.begin(); it != mLayers.end(); it++, index++)
    {
        bool found;
        list<HiShapeItem>::iterator match = (*it).findShape(name, found);
        if (found)
        {
            (*it).removeShape(match);
            // promote the match to the
            addShapeAtIndex(*match, index+1);

            return;
        }
    }
    
}


