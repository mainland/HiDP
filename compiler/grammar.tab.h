/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

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
/* Line 1529 of yacc.c.  */
#line 338 "grammar.tab.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

