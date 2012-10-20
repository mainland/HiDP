#pragma once 

/* Each statement node is associated with a Shape 
 * The shape corresponds to the parallelism level of this statement
 * For statement outside a map clause, the shape is the same size as any of the input or output
 * For statement inside a map clause, the shape is 1
 */

class StatementParserNode;

class StatementShape {
public:
    explicit StatementShape(StatementParserNode *parent) :
        parentStatement(parent)
    {}
    
    bool operator==(const StatementShape &rhs) const;
    
private:
    StatementParserNode *parentStatement;
    
    
};