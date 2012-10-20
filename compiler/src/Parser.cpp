#include "Parser.h"
#include "ParserNodes.h"
#include "StatementNodes.h"
#include <iostream>
using namespace std;

void ParserNode::setNext(ParserNode *next)
{
  mNext = next;
}

void ParserNode::hprint_w(int level)
{
  printLevel(level);
  hprint(level);
  printNext(level);
}

void ParserNode::printLevel(int level) const
{
  for (int i = 0; i < level; i++)
    cout << "---";
}

void ParserNode::printNext(int level)
{
  if (mNext)
    mNext->hprint_w(level);
}

bool StatementParserNode::findSymbol(std::string &symbol, int &blockid) const
{
  if (mParentBlock)
    return mParentBlock->findSymbol(symbol, blockid);
  return false;
}



/*
void StatementParserNode::printInOut() const
{
  cout << "input (";
  for (InoutType::const_iterator it = mInputs.begin(); it != mInputs.end(); it++)
    {
      cout << (*it).first << " ";
    }
  cout << ") ";
  cout << "output (";
  for (InoutType::const_iterator it = mOutputs.begin(); it !=  mOutputs.end(); it++)
    {
      cout << (*it).first << " ";
    }
  cout << ")";
  cout << "inout (";
  for (InoutType::const_iterator it = mInOuts.begin(); it != mInOuts.end(); it++)
    {
      cout << (*it).first << " ";
    }
  cout << ")" ;

}
*/


