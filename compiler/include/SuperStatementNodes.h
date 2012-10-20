#pragma once
#include "Parser.h"
#include <vector>


class SuperStatement: public StatementParserNode
{
public:
    SuperStatement():StatementParserNode(Super, false) 
    {}
    ~SuperStatement() {}
    
    void addStatement(StatementParserNode *stmt);
    
    virtual void analyzeInOut() {}
    
    int getNumStatements() const 
    {
        return (int)mStatements.size();
    }
    virtual void run() {}
    
    virtual void hprint(int level);  
    virtual void setStatementBelow() {  assert(0); // shouldn't be here
    }
    
    bool compatible(StatementParserNode *);
    HiShapeLayer getShape() const { return mShape; }
    bool mapExecution(boost::shared_ptr<class HierarchyMode> &mode);
private:
    
    std::vector<StatementParserNode *> mStatements;
    HiShapeLayer mShape;
    
    class Mapping{
    public:
        void addMapping(boost::shared_ptr<class HierarchyMode> m) { aMapping.push_back(m); }
    public:
        std::vector<boost::shared_ptr<class HierarchyMode> > aMapping;
    };
    
    std::vector<Mapping> mMappings;
    
    
};


