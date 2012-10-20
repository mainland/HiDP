/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     p_map = 258,
     FUNCTION = 259,
     p_sort = 260,
     p_scan = 261,
     p_reduce = 262,
     ASSIGN = 263,
     ITER_ASSIGN = 264,
     PLUS_ASSIGN = 265,
     MINUS_ASSIGN = 266,
     MULTI_ASSIGN = 267,
     DIV_ASSIGN = 268,
     MAP = 269,
     MINUS = 270,
     PLUS = 271,
     MAX = 272,
     MIN = 273,
     MOD = 274,
     DIV = 275,
     TIMES = 276,
     RSHIFT = 277,
     LSHIFT = 278,
     NEQ = 279,
     EQ = 280,
     GE = 281,
     GT = 282,
     LE = 283,
     LT = 284,
     XOR = 285,
     OR = 286,
     AND = 287,
     NOT = 288,
     CLOSEBLOCK = 289,
     OPENBLOCK = 290,
     CLOSEBRACKET = 291,
     OPENBRACKET = 292,
     CLOSEPAR = 293,
     OPENPAR = 294,
     SEMI = 295,
     COLON = 296,
     COMA = 297,
     SELECT = 298,
     RAND = 299,
     FLOOR = 300,
     CEIL = 301,
     TRUNC = 302,
     ROUND = 303,
     LOG = 304,
     SQRT = 305,
     EXP = 306,
     SIN = 307,
     COS = 308,
     TAN = 309,
     ASIN = 310,
     ACOS = 311,
     ATAN = 312,
     SINH = 313,
     COSH = 314,
     TANH = 315,
     I_TO_F = 316,
     I_TO_B = 317,
     B_TO_I = 318,
     PLUS_SCAN = 319,
     MULT_SCAN = 320,
     MAX_SCAN = 321,
     MIN_SCAN = 322,
     AND_SCAN = 323,
     OR_SCAN = 324,
     XOR_SCAN = 325,
     PLUS_REDUCE = 326,
     MULT_REDUCE = 327,
     MAX_REDUCE = 328,
     MIN_REDUCE = 329,
     AND_REDUCE = 330,
     OR_REDUCE = 331,
     XOR_REDUCE = 332,
     PERMUTE = 333,
     DPERMUTE = 334,
     FPERMUTE = 335,
     BPERMUTE = 336,
     BFPERMUTE = 337,
     DFPERMUTE = 338,
     EXTRACT = 339,
     REPLACE = 340,
     DIST = 341,
     INDEX = 342,
     LENGTH = 343,
     MAKE_SEGDES = 344,
     LENGTHS = 345,
     COPY = 346,
     POP = 347,
     CPOP = 348,
     PAIR = 349,
     UNPAIR = 350,
     CALL = 351,
     RET = 352,
     FUNC = 353,
     IF = 354,
     ELSE = 355,
     ENDIF = 356,
     CONST = 357,
     EXIT = 358,
     START_TIMER = 359,
     STOP_TIMER = 360,
     READ = 361,
     WRITE = 362,
     FOPEN = 363,
     FCLOSE = 364,
     FREAD = 365,
     FREAD_CHAR = 366,
     FWRITE = 367,
     SPAWN = 368,
     SRAND = 369,
     INT = 370,
     BOOL = 371,
     FLOAT = 372,
     SEGDES = 373,
     CHAR = 374,
     V_TRUE = 375,
     V_FALSE = 376,
     NULL_STREAM = 377,
     STDIN = 378,
     STDOUT = 379,
     STDERR = 380,
     MAIN = 381,
     BEGIN_VECTOR = 382,
     END_VECTOR = 383,
     INTEGER = 384,
     REAL = 385,
     ASTRING = 386,
     IDENTIFIER = 387,
     INPUT_INFO = 388,
     OUTPUT_INFO = 389,
     INPUT = 390,
     OUTPUT = 391,
     INOUT = 392
   };
#endif
/* Tokens.  */
#define p_map 258
#define FUNCTION 259
#define p_sort 260
#define p_scan 261
#define p_reduce 262
#define ASSIGN 263
#define ITER_ASSIGN 264
#define PLUS_ASSIGN 265
#define MINUS_ASSIGN 266
#define MULTI_ASSIGN 267
#define DIV_ASSIGN 268
#define MAP 269
#define MINUS 270
#define PLUS 271
#define MAX 272
#define MIN 273
#define MOD 274
#define DIV 275
#define TIMES 276
#define RSHIFT 277
#define LSHIFT 278
#define NEQ 279
#define EQ 280
#define GE 281
#define GT 282
#define LE 283
#define LT 284
#define XOR 285
#define OR 286
#define AND 287
#define NOT 288
#define CLOSEBLOCK 289
#define OPENBLOCK 290
#define CLOSEBRACKET 291
#define OPENBRACKET 292
#define CLOSEPAR 293
#define OPENPAR 294
#define SEMI 295
#define COLON 296
#define COMA 297
#define SELECT 298
#define RAND 299
#define FLOOR 300
#define CEIL 301
#define TRUNC 302
#define ROUND 303
#define LOG 304
#define SQRT 305
#define EXP 306
#define SIN 307
#define COS 308
#define TAN 309
#define ASIN 310
#define ACOS 311
#define ATAN 312
#define SINH 313
#define COSH 314
#define TANH 315
#define I_TO_F 316
#define I_TO_B 317
#define B_TO_I 318
#define PLUS_SCAN 319
#define MULT_SCAN 320
#define MAX_SCAN 321
#define MIN_SCAN 322
#define AND_SCAN 323
#define OR_SCAN 324
#define XOR_SCAN 325
#define PLUS_REDUCE 326
#define MULT_REDUCE 327
#define MAX_REDUCE 328
#define MIN_REDUCE 329
#define AND_REDUCE 330
#define OR_REDUCE 331
#define XOR_REDUCE 332
#define PERMUTE 333
#define DPERMUTE 334
#define FPERMUTE 335
#define BPERMUTE 336
#define BFPERMUTE 337
#define DFPERMUTE 338
#define EXTRACT 339
#define REPLACE 340
#define DIST 341
#define INDEX 342
#define LENGTH 343
#define MAKE_SEGDES 344
#define LENGTHS 345
#define COPY 346
#define POP 347
#define CPOP 348
#define PAIR 349
#define UNPAIR 350
#define CALL 351
#define RET 352
#define FUNC 353
#define IF 354
#define ELSE 355
#define ENDIF 356
#define CONST 357
#define EXIT 358
#define START_TIMER 359
#define STOP_TIMER 360
#define READ 361
#define WRITE 362
#define FOPEN 363
#define FCLOSE 364
#define FREAD 365
#define FREAD_CHAR 366
#define FWRITE 367
#define SPAWN 368
#define SRAND 369
#define INT 370
#define BOOL 371
#define FLOAT 372
#define SEGDES 373
#define CHAR 374
#define V_TRUE 375
#define V_FALSE 376
#define NULL_STREAM 377
#define STDIN 378
#define STDOUT 379
#define STDERR 380
#define MAIN 381
#define BEGIN_VECTOR 382
#define END_VECTOR 383
#define INTEGER 384
#define REAL 385
#define ASTRING 386
#define IDENTIFIER 387
#define INPUT_INFO 388
#define OUTPUT_INFO 389
#define INPUT 390
#define OUTPUT 391
#define INOUT 392




/* Copy the first part of user declarations.  */
#line 1 "grammar.y"
 
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


/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 33 "grammar.y"
{
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
/* Line 193 of yacc.c.  */
#line 413 "grammar.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 426 "grammar.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  6
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   221

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  138
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  23
/* YYNRULES -- Number of rules.  */
#define YYNRULES  64
/* YYNRULES -- Number of states.  */
#define YYNSTATES  117

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   392

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint8 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    10,    16,    17,    20,
      23,    26,    29,    33,    36,    40,    43,    47,    51,    56,
      60,    64,    69,    70,    75,    79,    80,    82,    84,    86,
      88,    89,    93,    96,    99,   103,   107,   111,   114,   118,
     122,   126,   130,   134,   136,   139,   143,   145,   147,   149,
     153,   157,   161,   165,   169,   173,   177,   181,   185,   189,
     193,   195,   199,   203,   207
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     139,     0,    -1,   140,    -1,    -1,   141,   140,    -1,    -1,
       4,   132,   142,   143,   149,    -1,    -1,   143,   144,    -1,
     143,   145,    -1,   143,   146,    -1,   135,   147,    -1,   144,
      42,   147,    -1,   136,   147,    -1,   145,    42,   147,    -1,
     137,   147,    -1,   146,    42,   147,    -1,   132,    41,   132,
      -1,   132,   148,    41,   132,    -1,   132,     9,   160,    -1,
      37,   158,    36,    -1,   148,    37,   158,    36,    -1,    -1,
      35,   150,   151,    34,    -1,   152,    40,   151,    -1,    -1,
     157,    -1,   147,    -1,   153,    -1,   156,    -1,    -1,    14,
     154,   149,    -1,   153,   156,    -1,   132,    39,    -1,   132,
      39,   158,    -1,   155,    42,   158,    -1,   155,    42,   147,
      -1,   155,    38,    -1,   158,     8,   158,    -1,   158,    10,
     158,    -1,   158,    11,   158,    -1,   158,    12,   158,    -1,
     158,    13,   158,    -1,   132,    -1,   132,   148,    -1,    39,
     158,    38,    -1,   129,    -1,   117,    -1,   131,    -1,   158,
      16,   158,    -1,   158,    15,   158,    -1,   158,    21,   158,
      -1,   158,    20,   158,    -1,   158,    25,   158,    -1,   158,
      28,   158,    -1,   158,    26,   158,    -1,   158,    27,   158,
      -1,   158,    29,   158,    -1,   158,    41,   158,    -1,   159,
      41,   158,    -1,    21,    -1,    37,   159,    36,    -1,    37,
     159,    38,    -1,    39,   159,    38,    -1,    39,   159,    36,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   121,   121,   131,   132,   136,   136,   148,   150,   154,
     158,   164,   171,   181,   188,   198,   205,   215,   228,   241,
     249,   253,   265,   265,   284,   297,   300,   307,   314,   322,
     331,   331,   337,   346,   349,   354,   359,   366,   371,   375,
     378,   381,   384,   389,   394,   399,   402,   407,   411,   415,
     419,   423,   426,   429,   432,   435,   438,   441,   446,   449,
     454,   462,   468,   474,   480
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "p_map", "FUNCTION", "p_sort", "p_scan",
  "p_reduce", "ASSIGN", "ITER_ASSIGN", "PLUS_ASSIGN", "MINUS_ASSIGN",
  "MULTI_ASSIGN", "DIV_ASSIGN", "MAP", "MINUS", "PLUS", "MAX", "MIN",
  "MOD", "DIV", "TIMES", "RSHIFT", "LSHIFT", "NEQ", "EQ", "GE", "GT", "LE",
  "LT", "XOR", "OR", "AND", "NOT", "CLOSEBLOCK", "OPENBLOCK",
  "CLOSEBRACKET", "OPENBRACKET", "CLOSEPAR", "OPENPAR", "SEMI", "COLON",
  "COMA", "SELECT", "RAND", "FLOOR", "CEIL", "TRUNC", "ROUND", "LOG",
  "SQRT", "EXP", "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN", "SINH",
  "COSH", "TANH", "I_TO_F", "I_TO_B", "B_TO_I", "PLUS_SCAN", "MULT_SCAN",
  "MAX_SCAN", "MIN_SCAN", "AND_SCAN", "OR_SCAN", "XOR_SCAN", "PLUS_REDUCE",
  "MULT_REDUCE", "MAX_REDUCE", "MIN_REDUCE", "AND_REDUCE", "OR_REDUCE",
  "XOR_REDUCE", "PERMUTE", "DPERMUTE", "FPERMUTE", "BPERMUTE", "BFPERMUTE",
  "DFPERMUTE", "EXTRACT", "REPLACE", "DIST", "INDEX", "LENGTH",
  "MAKE_SEGDES", "LENGTHS", "COPY", "POP", "CPOP", "PAIR", "UNPAIR",
  "CALL", "RET", "FUNC", "IF", "ELSE", "ENDIF", "CONST", "EXIT",
  "START_TIMER", "STOP_TIMER", "READ", "WRITE", "FOPEN", "FCLOSE", "FREAD",
  "FREAD_CHAR", "FWRITE", "SPAWN", "SRAND", "INT", "BOOL", "FLOAT",
  "SEGDES", "CHAR", "V_TRUE", "V_FALSE", "NULL_STREAM", "STDIN", "STDOUT",
  "STDERR", "MAIN", "BEGIN_VECTOR", "END_VECTOR", "INTEGER", "REAL",
  "ASTRING", "IDENTIFIER", "INPUT_INFO", "OUTPUT_INFO", "INPUT", "OUTPUT",
  "INOUT", "$accept", "program", "functions", "function", "@1",
  "function_prefix", "inputs", "outputs", "inouts", "definition",
  "brackets", "block", "@2", "statements", "statement", "map_block", "@3",
  "call_function_prefix", "call_function", "assignment", "aexpr",
  "range_internal", "range", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   138,   139,   140,   140,   142,   141,   143,   143,   143,
     143,   144,   144,   145,   145,   146,   146,   147,   147,   147,
     148,   148,   150,   149,   151,   151,   152,   152,   152,   152,
     154,   153,   153,   155,   155,   155,   155,   156,   157,   157,
     157,   157,   157,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   159,   159,
     159,   160,   160,   160,   160
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     5,     0,     2,     2,
       2,     2,     3,     2,     3,     2,     3,     3,     4,     3,
       3,     4,     0,     4,     3,     0,     1,     1,     1,     1,
       0,     3,     2,     2,     3,     3,     3,     2,     3,     3,
       3,     3,     3,     1,     2,     3,     1,     1,     1,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       1,     3,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       3,     0,     0,     2,     3,     5,     1,     4,     7,     0,
      22,     0,     0,     0,     8,     9,    10,     6,    25,     0,
      11,    13,    15,     0,     0,     0,    30,     0,    47,    46,
      48,    43,    27,     0,     0,    28,     0,    29,    26,     0,
       0,     0,     0,     0,    12,    14,    16,     0,    43,     0,
      33,    44,    23,    25,     0,    32,    37,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    19,     0,    17,     0,     0,    31,
      44,    45,    34,    24,    43,    36,    35,    38,    39,    40,
      41,    42,    50,    49,    52,    51,    53,    55,    56,    54,
      57,    60,     0,     0,     0,    20,     0,    18,     0,    61,
      62,     0,    64,    63,    21,    58,    59
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     2,     3,     4,     8,     9,    14,    15,    16,    32,
      51,    17,    18,    33,    34,    35,    47,    36,    37,    38,
      39,   103,    74
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -122
static const yytype_int16 yypact[] =
{
       1,  -121,    12,  -122,     1,  -122,  -122,  -122,  -122,   -31,
    -122,  -112,  -112,  -112,    -8,    -5,    -4,  -122,   -14,    20,
    -122,  -122,  -122,  -112,  -112,  -112,  -122,   -37,  -122,  -122,
    -122,    -6,  -122,   -10,    25,  -102,   -21,  -122,  -122,   128,
      19,   -37,   -62,   -19,  -122,  -122,  -122,    36,    40,   104,
     -37,   -19,  -122,   -14,    39,  -122,  -122,   -33,   -37,   -37,
     -37,   -37,   -37,   -37,   -37,   -37,   -37,   -37,   -37,   -37,
     -37,   -37,   -20,   -20,  -122,   143,  -122,   -37,   -53,  -122,
      44,  -122,   177,  -122,    20,  -122,   177,   177,   177,   177,
     177,   177,   187,   187,   192,   192,  -122,  -122,  -122,  -122,
    -122,  -122,    47,   -28,    28,  -122,   160,  -122,   -37,  -122,
    -122,   -37,  -122,  -122,  -122,   177,   177
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
    -122,  -122,    78,  -122,  -122,  -122,  -122,  -122,  -122,     3,
     -12,    38,  -122,    30,  -122,  -122,  -122,  -122,    51,  -122,
     -18,    14,  -122
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint8 yytable[] =
{
      26,   101,    27,    40,    10,     1,    27,    43,   109,    49,
     110,     5,     6,   111,    20,    21,    22,    56,    77,    27,
      19,    57,    78,    75,    52,    27,    44,    45,    46,    40,
      54,    41,    82,    50,    23,    42,    80,    24,    25,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   102,   102,    72,    41,    73,   106,
      85,    42,    63,    64,   112,    53,   113,    65,    66,   111,
      76,    10,    67,    68,    69,    70,    71,    41,    50,   107,
      28,    77,     7,    83,    28,    79,    55,   104,   108,     0,
     115,     0,    29,   116,    30,    48,    29,    28,    30,    84,
       0,     0,     0,    28,    11,    12,    13,     0,     0,    29,
       0,    30,    48,     0,     0,    29,     0,    30,    31,    63,
      64,     0,     0,     0,    65,    66,     0,     0,     0,    67,
      68,    69,    70,    71,     0,     0,    58,     0,    59,    60,
      61,    62,    81,    63,    64,     0,     0,     0,    65,    66,
       0,     0,     0,    67,    68,    69,    70,    71,    63,    64,
       0,     0,     0,    65,    66,     0,     0,     0,    67,    68,
      69,    70,    71,     0,     0,    63,    64,     0,     0,   105,
      65,    66,     0,     0,     0,    67,    68,    69,    70,    71,
       0,     0,    63,    64,     0,     0,   114,    65,    66,     0,
       0,     0,    67,    68,    69,    70,    71,    65,    66,     0,
       0,     0,    67,    68,    69,    70,    71,    67,    68,    69,
      70,    71
};

static const yytype_int16 yycheck[] =
{
      14,    21,    39,     9,    35,     4,    39,    19,    36,    27,
      38,   132,     0,    41,    11,    12,    13,    38,    37,    39,
     132,    42,    41,    41,    34,    39,    23,    24,    25,     9,
     132,    37,    50,    39,    42,    41,    48,    42,    42,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    37,    37,    39,    77,
      57,    41,    15,    16,    36,    40,    38,    20,    21,    41,
     132,    35,    25,    26,    27,    28,    29,    37,    39,   132,
     117,    37,     4,    53,   117,    47,    35,    73,    41,    -1,
     108,    -1,   129,   111,   131,   132,   129,   117,   131,   132,
      -1,    -1,    -1,   117,   135,   136,   137,    -1,    -1,   129,
      -1,   131,   132,    -1,    -1,   129,    -1,   131,   132,    15,
      16,    -1,    -1,    -1,    20,    21,    -1,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    -1,     8,    -1,    10,    11,
      12,    13,    38,    15,    16,    -1,    -1,    -1,    20,    21,
      -1,    -1,    -1,    25,    26,    27,    28,    29,    15,    16,
      -1,    -1,    -1,    20,    21,    -1,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    -1,    15,    16,    -1,    -1,    36,
      20,    21,    -1,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    -1,    15,    16,    -1,    -1,    36,    20,    21,    -1,
      -1,    -1,    25,    26,    27,    28,    29,    20,    21,    -1,
      -1,    -1,    25,    26,    27,    28,    29,    25,    26,    27,
      28,    29
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     4,   139,   140,   141,   132,     0,   140,   142,   143,
      35,   135,   136,   137,   144,   145,   146,   149,   150,   132,
     147,   147,   147,    42,    42,    42,    14,    39,   117,   129,
     131,   132,   147,   151,   152,   153,   155,   156,   157,   158,
       9,    37,    41,   148,   147,   147,   147,   154,   132,   158,
      39,   148,    34,    40,   132,   156,    38,    42,     8,    10,
      11,    12,    13,    15,    16,    20,    21,    25,    26,    27,
      28,    29,    37,    39,   160,   158,   132,    37,    41,   149,
     148,    38,   158,   151,   132,   147,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,    21,   158,   159,   159,    36,   158,   132,    41,    36,
      38,    41,    36,    38,    36,   158,   158
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 122 "grammar.y"
    {
  // Node *node = (Node *)$$; 
  // node->backEnd();
  // node->genCode();
  // delete node;
  printf("gen programs there are %d functions.\n", gFunctions.size());
;}
    break;

  case 3:
#line 131 "grammar.y"
    {;}
    break;

  case 4:
#line 133 "grammar.y"
    {
;}
    break;

  case 5:
#line 136 "grammar.y"
    { curFunction = new FuncDefParserNode((yyvsp[(2) - (2)].id));  ;}
    break;

  case 6:
#line 137 "grammar.y"
    {
  printf("generate function %s.\n", (yyvsp[(2) - (5)].id));
  (yyval.node) = curFunction;
  curFunction->setFuncBlock(dynamic_cast<BlockParserNode *>((yyvsp[(5) - (5)].block_node)));
  curFunction->genCodeArgument();
  gFunctions.insert(make_pair<string, FuncDefParserNode *>(curFunction->getFuncName(), curFunction));
 
;}
    break;

  case 7:
#line 148 "grammar.y"
    {
;}
    break;

  case 8:
#line 151 "grammar.y"
    {
  printf("generate function prefix w. input.\n");
;}
    break;

  case 9:
#line 155 "grammar.y"
    {
  (yyval.node) = NULL;
;}
    break;

  case 10:
#line 159 "grammar.y"
    {
  printf("generate function prefix w. inout.\n");
  (yyval.node) = NULL;
;}
    break;

  case 11:
#line 165 "grammar.y"
    {
  DefParserNode *node = dynamic_cast<DefParserNode *>((yyvsp[(2) - (2)].stmt_node));
  node->setScope(DefParserNode::Func_input);
  curFunction->addInput(node);
  (yyval.node) = node;
  //  printf("generating input 1.\n");
;}
    break;

  case 12:
#line 172 "grammar.y"
    {
  DefParserNode *node = dynamic_cast<DefParserNode *>((yyvsp[(3) - (3)].stmt_node));
  node->setScope(DefParserNode::Func_input);
  curFunction->addInput(node);
  (yyvsp[(1) - (3)].node)->setNext((yyvsp[(3) - (3)].stmt_node));
  (yyval.node) = (yyvsp[(1) - (3)].node);
  //  printf("generating input 2.\n");
;}
    break;

  case 13:
#line 182 "grammar.y"
    {
  DefParserNode *node = dynamic_cast<DefParserNode *>((yyvsp[(2) - (2)].stmt_node));
  node->setScope(DefParserNode::Func_output);
  curFunction->addOutput(node);
  printf("generating output 1.\n");
  (yyval.node) = node;
;}
    break;

  case 14:
#line 189 "grammar.y"
    {
  DefParserNode *node = dynamic_cast<DefParserNode *>((yyvsp[(3) - (3)].stmt_node));
  node->setScope(DefParserNode::Func_output);
  curFunction->addOutput(node);
  (yyvsp[(1) - (3)].node)->setNext((yyvsp[(3) - (3)].stmt_node));
  (yyval.node) = (yyvsp[(1) - (3)].node);
  printf("generating output 2.\n");
;}
    break;

  case 15:
#line 199 "grammar.y"
    {
  DefParserNode *node = dynamic_cast<DefParserNode *>((yyvsp[(2) - (2)].stmt_node));
  node->setScope(DefParserNode::Func_inout);
  curFunction->addInout(node);
  (yyval.node) = node;
  printf("generating inout 1.\n");
;}
    break;

  case 16:
#line 206 "grammar.y"
    {
  DefParserNode *node = dynamic_cast<DefParserNode *>((yyvsp[(3) - (3)].stmt_node));
  node->setScope(DefParserNode::Func_inout);
  curFunction->addInout(node);
  (yyvsp[(1) - (3)].node)->setNext((yyvsp[(3) - (3)].stmt_node));
  (yyval.node) = (yyvsp[(1) - (3)].node);
  printf("generating inout 2.\n");
;}
    break;

  case 17:
#line 216 "grammar.y"
    { 
  // scalar definition
  DefParserNode *def = new DefParserNode((yyvsp[(1) - (3)].id), (yyvsp[(3) - (3)].id), curMapLevel > 0);  // a scalar
  printf("blockstack size %d.\n", blockStack.size());
  if (blockStack.size() > 0)
    {
      printf("adding symbol.(%s)..\n", def->getName().c_str());
      //      def->hprint_w(0);
      blockStack[blockStack.size()-1]->addSymbol(def);
    }
  (yyval.stmt_node) = def; 
  printf("generate scalar definition type %s.\n", (yyvsp[(3) - (3)].id));
;}
    break;

  case 18:
#line 229 "grammar.y"
    {
  DefParserNode *def = new DefParserNode((yyvsp[(1) - (4)].id), dynamic_cast<IndexParserNode *>((yyvsp[(2) - (4)].node)), (yyvsp[(4) - (4)].id), curMapLevel > 0);  // an array
  (yyval.stmt_node) = def;
  printf("blockstack size %d.\n", blockStack.size());
  if (blockStack.size() > 0)
    {
      printf("adding symbol.(%s)..\n", def->getName().c_str());
      //      def->hprint_w(0);
      blockStack[blockStack.size()-1]->addSymbol(def);
    }
   // array definition
  printf("generate array definition type %s.\n", (yyvsp[(4) - (4)].id));
;}
    break;

  case 19:
#line 241 "grammar.y"
    {
  assert(blockStack.size() > 0);
  DefParserNode *defNode = new DefParserNode((yyvsp[(1) - (3)].id), dynamic_cast<RangeParserNode *>((yyvsp[(3) - (3)].node)), curMapLevel > 0);
  blockStack[blockStack.size()-1]->addIterVars(defNode);
  (yyval.stmt_node) = defNode;
  //  $$ = new OpParserNode(rNode($1), $3, ":=");
;}
    break;

  case 20:
#line 250 "grammar.y"
    {
  (yyval.node) = new IndexParserNode((yyvsp[(2) - (3)].node)); // 
  
;}
    break;

  case 21:
#line 254 "grammar.y"
    {
  IndexParserNode *node = dynamic_cast<IndexParserNode *>((yyvsp[(1) - (4)].node));
  node->shift((yyvsp[(3) - (4)].node));
  (yyval.node) = node;
;}
    break;

  case 22:
#line 265 "grammar.y"
    { curBlockLevel++;  
  BlockParserNode *parent = blockStack.size() > 0? blockStack[blockStack.size()-1] : NULL;
  BlockParserNode *blockn = new BlockParserNode(parent);
  if (parent)
    parent->appendChild(blockn);
  printf("pushing blockstack.\n");
  blockStack.push_back(blockn);   ;}
    break;

  case 23:
#line 273 "grammar.y"
    {
  assert(blockStack.size() > 0);
  BlockParserNode* blockn = blockStack[blockStack.size()-1];
  blockn->setStatements((yyvsp[(3) - (4)].stmt_node));
  (yyval.block_node) = blockn;

  blockStack.pop_back();
  curBlockLevel--;
  printf("poping blockstack.\n");
;}
    break;

  case 24:
#line 285 "grammar.y"
    {
  printf("generating statements.\n");
  assert((yyvsp[(1) - (3)].stmt_node) != 0);
  StatementParserNode *statement =  dynamic_cast<StatementParserNode *>((yyvsp[(1) - (3)].stmt_node));
  // pass definition statement
  if (statement->getType()  == StatementParserNode::Definition) 
    (yyval.stmt_node) = (yyvsp[(3) - (3)].stmt_node);
  else 
    {
      (yyvsp[(1) - (3)].stmt_node)->setNext((yyvsp[(3) - (3)].stmt_node));
      (yyval.stmt_node) = (yyvsp[(1) - (3)].stmt_node);
    }
;}
    break;

  case 25:
#line 297 "grammar.y"
    { (yyval.stmt_node) = NULL; ;}
    break;

  case 26:
#line 301 "grammar.y"
    {
  printf("generating statement.\n");
  (yyvsp[(1) - (1)].stmt_node)->setMapLevel(curMapLevel);
  (yyval.stmt_node) = (yyvsp[(1) - (1)].stmt_node);
  (yyval.stmt_node)->setStatementBelow();
  (yyval.stmt_node)->setParentBlock(blockStack[blockStack.size()-1]);
;}
    break;

  case 27:
#line 308 "grammar.y"
    {
  (yyvsp[(1) - (1)].stmt_node)->setMapLevel(curMapLevel);
  // add to the symbol table of the current block
  (yyval.stmt_node) = (yyvsp[(1) - (1)].stmt_node);
  (yyval.stmt_node)->setStatementBelow();
  (yyval.stmt_node)->setParentBlock(blockStack[blockStack.size()-1]);
;}
    break;

  case 28:
#line 315 "grammar.y"
    {
  (yyvsp[(1) - (1)].stmt_node)->setMapLevel(curMapLevel);
  (yyval.stmt_node) = (yyvsp[(1) - (1)].stmt_node);
  //  $1->addSymbol(new DefParserNode($1, dynamic_cast<RangeParserNode *>($3), curMapLevel > 0)); // map iterators must be int 
  (yyval.stmt_node)->setStatementBelow();
  (yyval.stmt_node)->setParentBlock(blockStack[blockStack.size()-1]);
  printf("generating a statement from map block.\n");
;}
    break;

  case 29:
#line 323 "grammar.y"
    {
  //  assert(0);
  (yyvsp[(1) - (1)].stmt_node)->setMapLevel(curMapLevel);
  (yyval.stmt_node) = (yyvsp[(1) - (1)].stmt_node);
  (yyval.stmt_node)->setStatementBelow();
  (yyval.stmt_node)->setParentBlock(blockStack[blockStack.size()-1]);
;}
    break;

  case 30:
#line 331 "grammar.y"
    {curMapLevel = curMapLevel+1; ;}
    break;

  case 31:
#line 332 "grammar.y"
    {
  MapParserNode *mapn = new MapParserNode((yyvsp[(3) - (3)].block_node), curMapLevel, curMapLevel > 0);
  mapn->addIteratorToSymbolTable();
  (yyval.stmt_node) = mapn;
  curMapLevel--;
;}
    break;

  case 32:
#line 338 "grammar.y"
    {
  MapParserNode *tmp = dynamic_cast<MapParserNode *>((yyvsp[(1) - (2)].stmt_node));
  tmp->addIteratorToSymbolTable();
  tmp->addSuffix(dynamic_cast<FunctionParserNode *>((yyvsp[(2) - (2)].stmt_node)));
  (yyval.stmt_node) = tmp;
;}
    break;

  case 33:
#line 347 "grammar.y"
    {
  (yyval.stmt_node) = new FunctionParserNode((yyvsp[(1) - (2)].id), curMapLevel > 0);
;}
    break;

  case 34:
#line 350 "grammar.y"
    {
  FunctionParserNode *node = new FunctionParserNode((yyvsp[(1) - (3)].id), curMapLevel > 0);
  node->shiftArg((yyvsp[(3) - (3)].node));
  (yyval.stmt_node) = node;
;}
    break;

  case 35:
#line 355 "grammar.y"
    {
  FunctionParserNode *node = dynamic_cast<FunctionParserNode *>((yyvsp[(1) - (3)].stmt_node));
  node->shiftArg((yyvsp[(3) - (3)].node));
  (yyval.stmt_node) = node;
;}
    break;

  case 36:
#line 360 "grammar.y"
    {
  FunctionParserNode *node = dynamic_cast<FunctionParserNode *>((yyvsp[(1) - (3)].stmt_node));
  node->setRange((yyvsp[(3) - (3)].stmt_node));
  (yyval.stmt_node) = node;
;}
    break;

  case 37:
#line 367 "grammar.y"
    {
  (yyval.stmt_node) = (yyvsp[(1) - (2)].stmt_node);
;}
    break;

  case 38:
#line 371 "grammar.y"
    {
  printf("curMapLevel %d.\n", curMapLevel);
  (yyval.stmt_node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "=", curMapLevel > 0);
;}
    break;

  case 39:
#line 375 "grammar.y"
    {
  (yyval.stmt_node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "+=", curMapLevel > 0);
;}
    break;

  case 40:
#line 378 "grammar.y"
    {
  (yyval.stmt_node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "-=", curMapLevel > 0);
;}
    break;

  case 41:
#line 381 "grammar.y"
    {
  (yyval.stmt_node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "*=", curMapLevel > 0);
;}
    break;

  case 42:
#line 384 "grammar.y"
    {
  (yyval.stmt_node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "/=", curMapLevel > 0);
;}
    break;

  case 43:
#line 390 "grammar.y"
    {
  printf("generating a expr from %s.\n", (yyvsp[(1) - (1)].id));
  (yyval.node) = new IdParserNode((yyvsp[(1) - (1)].id));
;}
    break;

  case 44:
#line 395 "grammar.y"
    {
  IndexParserNode *node = dynamic_cast<IndexParserNode *>((yyvsp[(2) - (2)].node));
  node->setId(new IdParserNode((yyvsp[(1) - (2)].id)));
  (yyval.node) = node;
;}
    break;

  case 45:
#line 400 "grammar.y"
    {
  (yyval.node) = (yyvsp[(2) - (3)].node);
;}
    break;

  case 46:
#line 403 "grammar.y"
    {
//  printf("generating a expr.\n");
  (yyval.node) = new IntegerParserNode((yyvsp[(1) - (1)].ival));
;}
    break;

  case 47:
#line 408 "grammar.y"
    {
//  printf("generating a expr.\n");
    (yyval.node) = new FloatParserNode((yyvsp[(1) - (1)].vop));
;}
    break;

  case 48:
#line 412 "grammar.y"
    {
  (yyval.node) = new StringParserNode((yyvsp[(1) - (1)].cval));
  printf("generating a expr from string.\n");
;}
    break;

  case 49:
#line 416 "grammar.y"
    {
    (yyval.node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "+", curMapLevel > 0);
//  printf("generating a expr.\n");
;}
    break;

  case 50:
#line 420 "grammar.y"
    {
    (yyval.node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "-", curMapLevel > 0);
  printf("generating a expr.\n");
;}
    break;

  case 51:
#line 424 "grammar.y"
    {
    (yyval.node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "*", curMapLevel > 0);
;}
    break;

  case 52:
#line 427 "grammar.y"
    {
    (yyval.node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "/", curMapLevel > 0);
;}
    break;

  case 53:
#line 430 "grammar.y"
    {
    (yyval.node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "==", curMapLevel > 0);
;}
    break;

  case 54:
#line 433 "grammar.y"
    {
    (yyval.node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "<=", curMapLevel > 0);
;}
    break;

  case 55:
#line 436 "grammar.y"
    {
    (yyval.node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), ">=", curMapLevel > 0);
;}
    break;

  case 56:
#line 439 "grammar.y"
    {
    (yyval.node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), ">", curMapLevel > 0);
;}
    break;

  case 57:
#line 442 "grammar.y"
    {
    (yyval.node) = new OpParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node), "<", curMapLevel > 0);
;}
    break;

  case 58:
#line 447 "grammar.y"
    {
    (yyval.node) = new RangeParserNode((yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
;}
    break;

  case 59:
#line 450 "grammar.y"
    {
  RangeParserNode *node = dynamic_cast<RangeParserNode *>((yyvsp[(1) - (3)].node));
  node->shift((yyvsp[(3) - (3)].node));
  (yyval.node) = node;
;}
    break;

  case 60:
#line 455 "grammar.y"
    { 
// wildcast
//  assert(0);
  (yyval.node) = new RangeParserNode();
  printf("generating a wildcast range internal.\n");
;}
    break;

  case 61:
#line 463 "grammar.y"
    {
  RangeParserNode *node = dynamic_cast<RangeParserNode *>((yyvsp[(2) - (3)].node));
  node->setLeftClose(true);
  node->setRightClose(true);
  (yyval.node) = node;
;}
    break;

  case 62:
#line 469 "grammar.y"
    {
  RangeParserNode *node = dynamic_cast<RangeParserNode *>((yyvsp[(2) - (3)].node));
  node->setLeftClose(true);
  node->setRightClose(false);
  (yyval.node) = node;
;}
    break;

  case 63:
#line 475 "grammar.y"
    {
  RangeParserNode *node = dynamic_cast<RangeParserNode *>((yyvsp[(2) - (3)].node));
  node->setLeftClose(false);
  node->setRightClose(false);
  (yyval.node) = node;
;}
    break;

  case 64:
#line 481 "grammar.y"
    {
  RangeParserNode *node = dynamic_cast<RangeParserNode *>((yyvsp[(2) - (3)].node));
  node->setLeftClose(false);
  node->setRightClose(true);
  (yyval.node) = node;
;}
    break;


/* Line 1267 of yacc.c.  */
#line 2352 "grammar.tab.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 488 "grammar.y"


void yyerror(char *info)
{
 printf("error ocurs for %s.\n", info);
 return;
}

