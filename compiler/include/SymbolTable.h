#pragma once

//#include "Parser.h"
class DefParserNode;
class ParserNode;
#include <map>
#include <string>
#include "Dim.h"

struct SymbolType
{
    Dim dim;
    std::string typestr;
};

class BlockParserNode;

class SymbolTable{
 public:
  bool foundSymbol(const std::string &symbol) const;
//  void addSymbol(DefParserNode *def, BlockParserNode *belongBlock);
    
  void insertSymbol(const std::string &name, const SymbolType &type);

  int getSize() const 
  {
    //return (int)mTable.size();
      return (int)mSymbols.size();
  }

    void hprint(int level) const ;

 private:
  //  SymbolTable *mParent;
  //  std::map<std::string, DefParserNode *> mTable;
  std::map<std::string, SymbolType> mSymbols;
};
