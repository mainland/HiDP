#pragma once

#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>

class ParserNode;

class DimType{
public:
    virtual bool isConst() const { return false; }
    virtual bool isExpression() const { return false;}
    virtual bool isVar() const {return false; }
    bool operator==(const DimType &rhs) const;

private:
//    virtual bool isEqual(DimType &rhs) = 0;
};

class VarDim: public DimType{
public:
    VarDim(std::string &n, int id): DimType(), name(n), blockId(id) {}
    
    virtual bool isVar() const {return true;}
    std::string name;
    int blockId;

private:
    virtual bool isEqual(VarDim &rhs)
    {
        return (name == rhs.name && blockId == rhs.blockId);
    }
};

class IntDim: public DimType{
public:
    explicit IntDim(int v) : DimType(), val(v) {}
    virtual bool isConst() const {return true;}
   int val;
private:
    bool isEqual(IntDim &rhs) const
    {
        return (val == rhs.val);
    }
 
};

class OpParserNode;

class ExpressionDim: public DimType{
public:
    explicit ExpressionDim(OpParserNode *op) :DimType() { }
    virtual bool isExpression() const {return true;}
    boost::shared_ptr<OpParserNode> mOp;
private:
    bool isEqual(ExpressionDim &rhs) const
    {
//        return (val == rhs.val);
        assert(0); // TODO
        return false;
    }
    
};






/* represents a symbol's dimension and each dimension's value
 either constant or a variable in previous symbol table */
class Dim
{
public:
    
    Dim() {}
    
    bool operator==(const Dim &rhs);
    
    bool isScalar() const {return dims.size()==0; }
    bool isVector() const { return !isScalar(); }
    int getNumDims() const { return (int)dims.size(); }
    void pushDimension(boost::shared_ptr<DimType> dim) { dims.push_back(dim); }
    void setToScalar() { dims.clear(); }
private:

    std::vector<boost::shared_ptr<DimType> > dims;


};


