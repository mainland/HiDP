%{ 
/*
 yacc source file 
*/

#include <cstdio>
#include <string>
#include "include/global.h"
#include "include/ParserNodes.h"
#include <vector>
#include <string>
#include <map>
#include <cassert>
using namespace std;

 map<string, FuncDefParserNode *> gFunctions;
 FuncDefParserNode * curFunction;
 BlockParserNode *curBlock;
 BlockParserNode *curParentBlock(0);
 int curBlockLevel(0);
 vector<BlockParserNode*> blockStack;
 int curMapLevel(0);
void yyerror(char *s);
extern int yy_flex_debug;
int yylex(void);
// ParserNode;
%}



%start program

%union {
       int ival;
       double dval;
       char *cval;
       char *pfunction;
       char *id;
       int vop;
       char *keyword;

  struct ParserNode *node;
  struct StatementParserNode *stmt_node;
  struct BlockParserNode *block_node;
  
}

/* Elementwise operations */
%token <keyword> p_map FUNCTION
%token <pfunction> p_sort p_scan p_reduce

%token <vop> ASSIGN ITER_ASSIGN PLUS_ASSIGN MINUS_ASSIGN MULTI_ASSIGN DIV_ASSIGN MAP
%left <vop> PLUS MINUS 
%left <vop> TIMES DIV MOD MIN MAX
%left <vop> LT LE GT GE EQ NEQ LSHIFT RSHIFT
%left <vop> NOT AND OR XOR
%left <vop> OPENPAR CLOSEPAR OPENBRACKET CLOSEBRACKET OPENBLOCK CLOSEBLOCK
%token <vop> SEMI COLON COMA
%token <vop> SELECT RAND
%token <vop> FLOOR CEIL TRUNC ROUND 
%token <vop> LOG SQRT EXP
%token <vop> SIN COS TAN ASIN ACOS ATAN SINH COSH TANH
%token <vop> I_TO_F I_TO_B B_TO_I


/* Vector operations */
%token <vop> PLUS_SCAN MULT_SCAN MAX_SCAN MIN_SCAN AND_SCAN OR_SCAN XOR_SCAN
%token <vop> PLUS_REDUCE MULT_REDUCE MAX_REDUCE MIN_REDUCE 
%token <vop> AND_REDUCE OR_REDUCE XOR_REDUCE
%token <vop> PERMUTE DPERMUTE FPERMUTE BPERMUTE BFPERMUTE DFPERMUTE
%token <vop> EXTRACT REPLACE DIST INDEX

/* Segment descriptor operations */
%token <vop> LENGTH MAKE_SEGDES LENGTHS

/* Control operations */
%token <vop> COPY POP CPOP PAIR UNPAIR CALL RET FUNC IF ELSE ENDIF 
%token <vop> CONST
%token <vop> EXIT

/* timing ops */
%token <vop> START_TIMER STOP_TIMER

/* I/O operations */
%token <vop> READ WRITE FOPEN FCLOSE FREAD FREAD_CHAR FWRITE SPAWN

/* Seed random number generator */
%token <vop> SRAND

/* Types */
%token <vop> INT BOOL FLOAT SEGDES CHAR

/* Constants */
%token <ival> V_TRUE V_FALSE
%token <ival> NULL_STREAM STDIN STDOUT STDERR

/* Main */
%token <vop> MAIN

/* Miscellaneous */
%token <vop> BEGIN_VECTOR END_VECTOR
%token <ival> INTEGER 
%token <dval> REAL
%token <cval> ASTRING
%token <id> IDENTIFIER
%token <vop> INPUT_INFO OUTPUT_INFO INPUT OUTPUT INOUT


/*%type <vop> types_1 types_2 types_3 types_5 type_const
%type <ival> int_con bool_con file_con int_val bool_val 
%type <dval> float_val*/

%type <node> functions function aexpr function_prefix inputs  outputs inouts brackets   range_internal range 
%type <stmt_node> statements statement assignment definition map_block call_function  call_function_prefix
%type <block_node> block

 //%type <node> outputs inouts

%%				/* beginning of rules section */

program     :   functions
{
  // Node *node = (Node *)$$; 
  // node->backEnd();
  // node->genCode();
  // delete node;
  printf("gen programs there are %d functions.\n", gFunctions.size());
}
;
functions  :
{} 
| function  functions
{
}
        ;
function: FUNCTION IDENTIFIER { curFunction = new FuncDefParserNode($2);  } function_prefix block
{
  printf("generate function %s.\n", $2);
  $$ = curFunction;
  curFunction->setFuncBlock(dynamic_cast<BlockParserNode *>($5));
  curFunction->genCodeArgument();
  gFunctions.insert(make_pair<string, FuncDefParserNode *>(curFunction->getFuncName(), curFunction));
 
} 
;

function_prefix : 
{
} |
 function_prefix inputs
{
  printf("generate function prefix w. input.\n");
}
| function_prefix outputs
{
  $$ = NULL;
}
| function_prefix inouts
{
  printf("generate function prefix w. inout.\n");
  $$ = NULL;
}
;
inputs : INPUT definition 
{
  DefParserNode *node = dynamic_cast<DefParserNode *>($2);
  node->setScope(DefParserNode::Func_input);
  curFunction->addInput(node);
  $$ = node;
  //  printf("generating input 1.\n");
} | inputs COMA definition
{
  DefParserNode *node = dynamic_cast<DefParserNode *>($3);
  node->setScope(DefParserNode::Func_input);
  curFunction->addInput(node);
  $1->setNext($3);
  $$ = $1;
  //  printf("generating input 2.\n");
};

outputs : OUTPUT definition 
{
  DefParserNode *node = dynamic_cast<DefParserNode *>($2);
  node->setScope(DefParserNode::Func_output);
  curFunction->addOutput(node);
  printf("generating output 1.\n");
  $$ = node;
} | outputs COMA definition
{
  DefParserNode *node = dynamic_cast<DefParserNode *>($3);
  node->setScope(DefParserNode::Func_output);
  curFunction->addOutput(node);
  $1->setNext($3);
  $$ = $1;
  printf("generating output 2.\n");
};

inouts : INOUT definition 
{
  DefParserNode *node = dynamic_cast<DefParserNode *>($2);
  node->setScope(DefParserNode::Func_inout);
  curFunction->addInout(node);
  $$ = node;
  printf("generating inout 1.\n");
} | inouts COMA definition
{
  DefParserNode *node = dynamic_cast<DefParserNode *>($3);
  node->setScope(DefParserNode::Func_inout);
  curFunction->addInout(node);
  $1->setNext($3);
  $$ = $1;
  printf("generating inout 2.\n");
};

definition: IDENTIFIER COLON IDENTIFIER 
{ 
  // scalar definition
  DefParserNode *def = new DefParserNode($1, $3, curMapLevel > 0);  // a scalar
  printf("blockstack size %d.\n", blockStack.size());
  if (blockStack.size() > 0)
    {
      printf("adding symbol.(%s)..\n", def->getName().c_str());
      //      def->hprint_w(0);
      blockStack[blockStack.size()-1]->addSymbol(def);
    }
  $$ = def; 
  printf("generate scalar definition type %s.\n", $3);
}  |IDENTIFIER brackets COLON IDENTIFIER
{
  DefParserNode *def = new DefParserNode($1, dynamic_cast<IndexParserNode *>($2), $4, curMapLevel > 0);  // an array
  $$ = def;
  printf("blockstack size %d.\n", blockStack.size());
  if (blockStack.size() > 0)
    {
      printf("adding symbol.(%s)..\n", def->getName().c_str());
      //      def->hprint_w(0);
      blockStack[blockStack.size()-1]->addSymbol(def);
    }
   // array definition
  printf("generate array definition type %s.\n", $4);
} | IDENTIFIER ITER_ASSIGN range {
  assert(blockStack.size() > 0);
  DefParserNode *defNode = new DefParserNode($1, dynamic_cast<RangeParserNode *>($3), curMapLevel > 0);
  blockStack[blockStack.size()-1]->addIterVars(defNode);
  $$ = defNode;
  //  $$ = new OpParserNode(rNode($1), $3, ":=");
} ;

brackets: OPENBRACKET aexpr CLOSEBRACKET
{
  $$ = new IndexParserNode($2); // 
  
} | brackets OPENBRACKET aexpr CLOSEBRACKET
{
  IndexParserNode *node = dynamic_cast<IndexParserNode *>($1);
  node->shift($3);
  $$ = node;
};

/*bracket: 
{
  $$ = new IndexParserNode($2); // 
  };*/

block : OPENBLOCK { curBlockLevel++;  
  BlockParserNode *parent = blockStack.size() > 0? blockStack[blockStack.size()-1] : NULL;
  BlockParserNode *blockn = new BlockParserNode(parent);
  if (parent)
    parent->appendChild(blockn);
  printf("pushing blockstack.\n");
  blockStack.push_back(blockn);   } 
statements CLOSEBLOCK
{
  assert(blockStack.size() > 0);
  BlockParserNode* blockn = blockStack[blockStack.size()-1];
  blockn->setStatements($3);
  $$ = blockn;

  blockStack.pop_back();
  curBlockLevel--;
  printf("poping blockstack.\n");
}
;
statements: statement SEMI statements
{
  printf("generating statements.\n");
  assert($1 != 0);
  StatementParserNode *statement =  dynamic_cast<StatementParserNode *>($1);
  // pass definition statement
  if (statement->getType()  == StatementParserNode::Definition) 
    $$ = $3;
  else 
    {
      $1->setNext($3);
      $$ = $1;
    }
} | { $$ = NULL; } ;


statement : assignment
{
  printf("generating statement.\n");
  $1->setMapLevel(curMapLevel);
  $$ = $1;
  $$->setStatementBelow();
  $$->setParentBlock(blockStack[blockStack.size()-1]);
} | definition 
{
  $1->setMapLevel(curMapLevel);
  // add to the symbol table of the current block
  $$ = $1;
  $$->setStatementBelow();
  $$->setParentBlock(blockStack[blockStack.size()-1]);
}| map_block 
{
  $1->setMapLevel(curMapLevel);
  $$ = $1;
  //  $1->addSymbol(new DefParserNode($1, dynamic_cast<RangeParserNode *>($3), curMapLevel > 0)); // map iterators must be int 
  $$->setStatementBelow();
  $$->setParentBlock(blockStack[blockStack.size()-1]);
  printf("generating a statement from map block.\n");
} | call_function 
{
  //  assert(0);
  $1->setMapLevel(curMapLevel);
  $$ = $1;
  $$->setStatementBelow();
  $$->setParentBlock(blockStack[blockStack.size()-1]);
}
;
map_block: MAP {curMapLevel = curMapLevel+1; } block
{
  MapParserNode *mapn = new MapParserNode($3, curMapLevel, curMapLevel > 0);
  mapn->addIteratorToSymbolTable();
  $$ = mapn;
  curMapLevel--;
} | map_block call_function
{
  MapParserNode *tmp = dynamic_cast<MapParserNode *>($1);
  tmp->addIteratorToSymbolTable();
  tmp->addSuffix(dynamic_cast<FunctionParserNode *>($2));
  $$ = tmp;
}
;

call_function_prefix: IDENTIFIER OPENPAR 
{
  $$ = new FunctionParserNode($1, curMapLevel > 0);
} | IDENTIFIER OPENPAR aexpr
{
  FunctionParserNode *node = new FunctionParserNode($1, curMapLevel > 0);
  node->shiftArg($3);
  $$ = node;
} | call_function_prefix COMA aexpr
{
  FunctionParserNode *node = dynamic_cast<FunctionParserNode *>($1);
  node->shiftArg($3);
  $$ = node;
} | call_function_prefix COMA definition
{
  FunctionParserNode *node = dynamic_cast<FunctionParserNode *>($1);
  node->setRange($3);
  $$ = node;
};

call_function: call_function_prefix CLOSEPAR
{
  $$ = $1;
};

assignment :  aexpr ASSIGN aexpr{
  printf("curMapLevel %d.\n", curMapLevel);
  $$ = new OpParserNode($1, $3, "=", curMapLevel > 0);
}| 
aexpr PLUS_ASSIGN aexpr{
  $$ = new OpParserNode($1, $3, "+=", curMapLevel > 0);
} |
aexpr MINUS_ASSIGN aexpr{
  $$ = new OpParserNode($1, $3, "-=", curMapLevel > 0);
} |
aexpr MULTI_ASSIGN aexpr{
  $$ = new OpParserNode($1, $3, "*=", curMapLevel > 0);
} |
aexpr DIV_ASSIGN aexpr{
  $$ = new OpParserNode($1, $3, "/=", curMapLevel > 0);
}

;
aexpr : IDENTIFIER 
{
  printf("generating a expr from %s.\n", $1);
  $$ = new IdParserNode($1);
}  | 
IDENTIFIER  brackets
{
  IndexParserNode *node = dynamic_cast<IndexParserNode *>($2);
  node->setId(new IdParserNode($1));
  $$ = node;
} | OPENPAR aexpr CLOSEPAR
{
  $$ = $2;
} | INTEGER 
{
//  printf("generating a expr.\n");
  $$ = new IntegerParserNode($1);
}  |
FLOAT
{
//  printf("generating a expr.\n");
    $$ = new FloatParserNode($1);
}  | ASTRING
{
  $$ = new StringParserNode($1);
  printf("generating a expr from string.\n");
}  | aexpr PLUS aexpr 
{
    $$ = new OpParserNode($1, $3, "+", curMapLevel > 0);
//  printf("generating a expr.\n");
}  | aexpr MINUS aexpr 
{
    $$ = new OpParserNode($1, $3, "-", curMapLevel > 0);
  printf("generating a expr.\n");
}  | aexpr TIMES aexpr 
{
    $$ = new OpParserNode($1, $3, "*", curMapLevel > 0);
}  | aexpr DIV aexpr 
{
    $$ = new OpParserNode($1, $3, "/", curMapLevel > 0);
}  | aexpr EQ aexpr 
{
    $$ = new OpParserNode($1, $3, "==", curMapLevel > 0);
}  | aexpr LE aexpr 
{
    $$ = new OpParserNode($1, $3, "<=", curMapLevel > 0);
}  | aexpr GE aexpr 
{
    $$ = new OpParserNode($1, $3, ">=", curMapLevel > 0);
}  | aexpr GT aexpr 
{
    $$ = new OpParserNode($1, $3, ">", curMapLevel > 0);
} | aexpr LT aexpr 
{
    $$ = new OpParserNode($1, $3, "<", curMapLevel > 0);
};

range_internal: aexpr COLON aexpr
{
    $$ = new RangeParserNode($1, $3);
} | range_internal COLON aexpr
{
  RangeParserNode *node = dynamic_cast<RangeParserNode *>($1);
  node->shift($3);
  $$ = node;
} | TIMES
{ 
// wildcast
//  assert(0);
  $$ = new RangeParserNode();
  printf("generating a wildcast range internal.\n");
};

range: OPENBRACKET range_internal CLOSEBRACKET
{
  RangeParserNode *node = dynamic_cast<RangeParserNode *>($2);
  node->setLeftClose(true);
  node->setRightClose(true);
  $$ = node;
} | OPENBRACKET range_internal CLOSEPAR
{
  RangeParserNode *node = dynamic_cast<RangeParserNode *>($2);
  node->setLeftClose(true);
  node->setRightClose(false);
  $$ = node;
} | OPENPAR range_internal CLOSEPAR
{
  RangeParserNode *node = dynamic_cast<RangeParserNode *>($2);
  node->setLeftClose(false);
  node->setRightClose(false);
  $$ = node;
} | OPENPAR range_internal CLOSEBRACKET
{
  RangeParserNode *node = dynamic_cast<RangeParserNode *>($2);
  node->setLeftClose(false);
  node->setRightClose(true);
  $$ = node;
};

%%

void yyerror(char *info)
{
 printf("error ocurs for %s.\n", info);
 return;
}
