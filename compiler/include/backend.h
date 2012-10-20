#pragma once
#include "global.h"
#include "ParserNodes.h"
#include <map>
#include <string>
#include "HierarchyMode.h"

void fuseStatement(const Params & params, std::map<std::string, FuncDefParserNode *> &functions);


void  map_execution_mode(const Params & params, std::map<std::string, FuncDefParserNode *> &functions, HierarchyMode *execution_mode);

void shapeAnalysis(const Params & params, std::map<std::string, FuncDefParserNode *> &functions);