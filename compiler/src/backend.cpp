#include "backend.h"
#include <map>
#include "ParserNodes.h"

using namespace std;
void fuseStatement(const Params &params, map<string, FuncDefParserNode *> &functions)
{
  if (functions.size() > 1 || functions.size() == 0)
    throw;

  FuncDefParserNode *curFunction = (*functions.begin()).second;

  // backend
  curFunction->analyzeInOut();
  curFunction->fuseToSuperStatement(params.OptLevel);
  curFunction->hprint_w(0); 

  // 
}

void shapeAnalysis(const Params & params, std::map<std::string, FuncDefParserNode *> &functions)
{
    if (functions.size() > 1 || functions.size() == 0)
        throw;
    FuncDefParserNode *topFunction = (*functions.begin()).second;
    
    topFunction->shapeAnalysis();
}

/* 
  top-bottom mapping of function to execution mode
*/
void  map_execution_mode(const Params & params, std::map<std::string, FuncDefParserNode *> &functions, HierarchyMode *execution_mode)
{
    assert(functions.size() == 1); // TODO support multiple functions later
    
    
}
