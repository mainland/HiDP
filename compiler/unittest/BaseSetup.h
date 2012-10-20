#pragma once
#include "gtest/gtest.h"
#include <string>
#include <map>
#include "grammar.tab.hpp"
#include "ParserNodes.h"
#include "global.h"
#include "gpu_mode.h"
#include "backend.h"
#include <boost/shared_ptr.hpp>

//using ::testing;

extern int yyparse();
extern FILE *yyin;
struct yy_buffer_state;
extern yy_buffer_state *yy_scan_string(const char *);
void yy_delete_buffer (yy_buffer_state *b );
extern std::map<std::string, FuncDefParserNode *> gFunctions;

class CaseTest : public testing::Test {
public:
//    explicit CaseTest(std::string str):
//    mInputString(str)
//    {
//    
//    }
    virtual void setupString() = 0;
protected:
    virtual void SetUp()
    {
        setupString();
        yy_buffer_state *my_string_buffer = yy_scan_string(mInputString.c_str());
        if (! (yyparse() == 0 ))
        {
            fprintf(stderr, "parse error.\n");
        }

        yy_delete_buffer(my_string_buffer);
        
        
    }
    std::string mInputString;
    Params mParams;
};
