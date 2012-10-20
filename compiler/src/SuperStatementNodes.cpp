#include "Parser.h"
#include "ParserNodes.h"
#include "HierarchyMode.h"
#include <iostream>
using namespace std;
using namespace boost;

void SuperStatement::addStatement(StatementParserNode *stmt)
{
    mStatements.push_back(stmt);
    // add statement
    //  InOut
    //  mNext.reset(new SuperStatement())
    const InoutType &stmt_input = stmt->getInputs();
    const InoutType &stmt_output = stmt->getOutputs();
    const InoutType &stmt_inout = stmt->getInOuts();
    for (InoutType::const_iterator in = stmt_input.begin(); in != stmt_input.end(); in++)
    {
        if (mOutputs.find((*in).first) == mOutputs.end())    // if not in my output set, then it is an input
            mInputs.insert((*in));
    }
    
    for (InoutType::const_iterator out = stmt_output.begin(); out != stmt_output.end(); out++)
    {
        if (mInputs.find((*out).first) != mInputs.end())  // if it is already in the input list, make it an inout
        {
            mOutputs.erase((*out).first);
            mInOuts.insert((*out));
        }
        else 
            mOutputs.insert((*out));
    }
    
    for (InoutType::const_iterator inout = stmt_inout.begin(); inout != stmt_inout.end(); inout++)
    {
        mInOuts.insert((*inout));      
    }
    
    mShape.fuseShape(stmt->getShape());
    stmt->setSuperStatement(this);
    // TODO add function name
}


void SuperStatement::hprint(int level)
{
    assert(0);//
}


/* 
 when should we fuse superstatement with statement?
 when shapes are compatible
// when there is overlap between inputs
// when there is overlap between outputs
// when there is overlap between the statement's input and superstatement's outputs or inputs
 
// Must be done after shape analysis
 
 
 too optimistic in current implementation. need more checking later TODO
 */
bool SuperStatement::compatible(StatementParserNode *stmt)
{
    assert(stmt);
    
#if 0
    const InoutType &stmt_input = stmt->getInputs();
    const InoutType &stmt_output = stmt->getOutputs();
    const InoutType &stmt_inout = stmt->getInOuts();
    bool overlap = false;
    for (InoutType::const_iterator in = stmt_input.begin(); in != stmt_input.end(); in++)
    {
        if (mOutputs.find((*in).first) != mOutputs.end())    // if not in my output set, then it is an input
            overlap = true;
        if (mInputs.find((*in).first) != mInputs.end())
            overlap = true;
    }
    // TODO 
    if (overlap) return true;
    
    for (InoutType::const_iterator out = stmt_output.begin(); out != stmt_output.end(); out++)
    {
        if (mOutputs.find((*out).first) != mInputs.end())  // if it is already in the input list, make it an inout
        {
            mOutputs.erase((*out).first);
            mInOuts.insert((*out));
        }
        else 
            mOutputs.insert((*out));
    }
    if (overlap) return true;
    
    for (InoutType::const_iterator inout = stmt_inout.begin(); inout != stmt_inout.end(); inout++)
    {
        if (mOutputs.find((*inout).first) != mOutputs.end())    // if not in my output set, then it is an input
            overlap = true;
        if (mInputs.find((*inout).first) != mInputs.end())
            overlap = true;
    }
    return overlap;
#endif
    string myshape = getShape().emitFlatString();
    string stmtshape = stmt->getShape().emitFlatString();
    // detect whether one is the prefix of another
    return (myshape.find(stmtshape) && stmtshape.find(myshape)) ? false : true;
    
}


bool SuperStatement::mapExecution(shared_ptr<HierarchyMode> &mode)
{
    assert(mode.get());
    size_t num_model = 1 + mode->getNumBelow();
    size_t num_shapelayer = mShape.getNumLayers();
    assert(num_shapelayer > 0);
    
    // if just one layer, assign it to the last mode
    if (num_shapelayer == 1)
    {
        size_t steps = mode->getNumBelow();
        shared_ptr<HierarchyMode> cur = mode;
        while (steps--)
        {
            cur = cur->getBelow();
            assert(cur.get());
        }
        Mapping amapping;
        amapping.addMapping(cur);
        mMappings.push_back(amapping);
        return true;
    }
    
    // build candidate mappings
    std::vector<Mapping> mCandidates;
    for (size_t m = 0; m < num_model - 1; m++)
    {
        
    }
    
    Mapping amapping;
    shared_ptr<HierarchyMode> candidate;
    bool success = false;
    
    // start from the root mode
    shared_ptr<HierarchyMode> cur_mode = mode;

    for (size_t m = 0; m < num_model; m++)
    {
        shared_ptr<HierarchyMode> cur_mode1 = cur_mode;
        int cur_layer = 0;
        HiShape &s = mShape.getLayer(cur_layer);
//        if (s.isCompatible(cur_mode))
            
        
        
        cur_mode = cur_mode->getBelow();
    }
        
    for (int layer = 0; layer < num_shapelayer; layer++)
    {
        // TODO if (lay size is not constant
        if (!success) break;
    }
    
    if (success)
        mMappings.push_back(amapping);
    
//    int mShape.getNumLayers()
}
