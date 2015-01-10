/* A Bison parser, made by GNU Bison 2.7.12-4996.  */

/* Bison interface for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2013 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

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

#ifndef YY_YY_GRAMMAR_TAB_HPP_INCLUDED
# define YY_YY_GRAMMAR_TAB_HPP_INCLUDED
/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

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


#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{
/* Line 2053 of yacc.c  */
#line 32 "grammar.y"

       int ival;
       double dval;
       char *cval;
       char *pfunction;
       char *id;
       int vop;
       char *keyword;

  ParserNode *node;
  StatementParserNode *stmt_node;
  BlockParserNode *block_node;
  


/* Line 2053 of yacc.c  */
#line 210 "grammar.tab.hpp"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;

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

#endif /* !YY_YY_GRAMMAR_TAB_HPP_INCLUDED  */
