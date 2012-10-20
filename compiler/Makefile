LEX=flex
YACC=bison

ROOTDIR?=..

TARGET := cuNesl


LEX_YY_O = lex.yy.o
LEX_YY_C = lex.yy.cpp
LEXYA_L = lexya.lpp

YACC_TAB_O = grammar.tab.o
YACC_TAB_C = grammar.tab.cpp
YACC_TAB_H = grammar.tab.hpp
YACC_Y = grammar.y

USER_MAIN_FILE := main.cpp
USER_FILES := $(YACC_TAB_C) $(LEX_YY_C) 
USER_H_FILES := $(YACC_TAB_H)
compiler := 1
include ../Makefile

$(YACC_TAB_C) $(YACC_TAB_H): $(YACC_Y)
	$(YACC) -d --verbose --report=state -o $(YACC_TAB_C) $(YACC_Y)

$(LEX_YY_C):$(LEXYA_L)
	$(LEX) -o $(LEX_YY_C) $(LEXYA_L)

$(LEX_YY_O):$(LEX_YY_C)
	$(CC)  $(CPPFLAGS)  -c $(LEX_YY_C) -o $(LEX_YY_O)

