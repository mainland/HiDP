digit            [0-9]
integer          [-+]?{digit}+
float            [-+]?{digit}*\.{digit}+([eE][-+]?{digit}+)?
comment          #.*
identifier       [-+\_*/%<>=!.A-Za-z0-9]+
boolconst        [TF]
astring	     	\"[^"]*\"
whitespace       [ \t\n]



%{
#include <cstdlib>
#include "global.h"
#include "include/Parser.h"
#include "grammar.tab.h"
void yyerror(char *);
#define YYLMAX 256

%}

%%


":="    {printf("emit ITER_ASSIGN.\n"); return (ITER_ASSIGN); }
"="    { return (ASSIGN); }
"+="    { return (PLUS_ASSIGN); }
"-="    { return (MINUS_ASSIGN); }
"*="    { return (MULTI_ASSIGN); }
"/="    { return (DIV_ASSIGN); }
"+" { return (PLUS); }
"-" { return (MINUS); }
"*" { printf("emit times.\n"); return (TIMES); }
"/" { return (DIV); }
"MIN" { return (MIN); }
"MAX" { return (MAX); }
"==" { return (EQ); }
"<="    { return (LE); }
"<" { return (LT); }
">="    { return (GE); }
">" { return (GT); }
"(" { return (OPENPAR); }
")" { return (CLOSEPAR); }
"[" { return (OPENBRACKET); }
"]" { return (CLOSEBRACKET); }
"{" { return (OPENBLOCK); }
"}" { return (CLOSEBLOCK); }
";" { return (SEMI); }
":" { printf("emit COLON.\n"); return (COLON); }
"," { return (COMA); }
"input" { return (INPUT); }
"output" { return (OUTPUT); }
"inout" { return (INOUT); }
"function" { printf("emit FUNCTION.\n"); return (FUNCTION); }
"map" { printf("emit map.\n"); return (MAP); }


{float}        {
                   sscanf(yytext, "%lf", &yylval.dval);
#if DEBUG
                   printf("parser: received float %f.\n", yylval.dval);
#endif
                   return(FLOAT);
               }
{integer}      {
                   sscanf(yytext, "%d", &yylval.ival);
#if DEBUG
                   printf("parser: received integer %d.\n", yylval.ival);
#endif
                   return(INTEGER);
               }

{astring}       {
                   yylval.cval = strdup(yytext);
                   if (yyleng >= YYLMAX)
                   {
                       fprintf(stderr, "lexer line %d: string longer than %d characters, may not be proceessed properly.\n",
                       yylineno, YYLMAX);
                   } 
#if DEBUG
printf("parser: received string %s.\n", yylval.cval);
#endif

                   return(ASTRING);
               }
{identifier}   {
                   yylval.id = strdup(yytext);
#if DEBUG
printf("parser: received identifier %s.\n", yytext);
#endif
                   return(IDENTIFIER);
               }

{comment}        {
printf("found comment %s.\n", yytext);
      ;
    }
{whitespace}       ;
.              {
                   fprintf(stderr, "lexer parser: unexpected charater in input at line %d.\n", yylineno);
               }

%%

int yywrap(void)
{
return 1;
}
