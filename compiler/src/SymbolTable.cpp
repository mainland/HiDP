#include "SymbolTable.h"
#include "ParserNodes.h"
#include "StatementNodes.h"

using namespace std;
bool SymbolTable::foundSymbol(const string &symbol) const
{
  if (mSymbols.find(symbol) != mSymbols.end())
    return true;
  return false;
}

/*void SymbolTable::addSymbol(DefParserNode *def, BlockParserNode *belongBlock) 
{
    SymbolType type;
    type.typestr = def->getType();
    def->convertToDim(type.dim);
    mSymbols.insert(make_pair<string, SymbolType>(def->getName(), type));
//    assert(0);
//  mSymbols.insert(make_pair<string, DefParserNode*>(def->getName(), def));
  //  if (mTable.exists(symbol))
}*/

void SymbolTable::insertSymbol(const std::string &name, const SymbolType &type)
{
    mSymbols.insert(make_pair<string, SymbolType>(name, type));
}

void SymbolTable::hprint(int level) const 
{
  for (int i = 0; i < level; i++)
    cout << "---";
    for (map<string, SymbolType>::const_iterator it = mSymbols.begin(); it != mSymbols.end(); it++)
    {
        cout << (*it).first << " "; // TODO
    }
    cout << endl;
  /*for (map<string, DefParserNode *>::const_iterator it = mTable.begin(); it != mTable.end(); it++)
  {
    cout << (*it).first << " ";
  }*/
  cout << "\n";
  
}
