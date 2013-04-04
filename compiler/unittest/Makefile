# Copyright (C) 2009-2010 the North Carolina State University
#  This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

ROOTDIR := ../..

#TARGET := $(ROOTDIR)/compiler/bin/unittest_all
TARGET := ./unittest_all

USER_MAIN_FILE := main.cpp GemmUt.cpp  HimenoUt.cpp  StencilUt.cpp
#USER_CUDA_FILES := 
#USER_H_FILES := 
#USER_CUDA_FILES := all.cu

LEX_YY_O = ../lex.yy.o
LEX_YY_C = ../lex.yy.cpp
LEXYA_L = ../lexya.lpp

YACC_TAB_O = ../grammar.tab.o
YACC_TAB_C = ../grammar.tab.cpp
YACC_TAB_H = ../grammar.tab.hpp
YACC_Y = ../grammar.y

USER_FILES := $(YACC_TAB_C) $(LEX_YY_C) 
USER_H_FILES := $(YACC_TAB_H)

UNIT_TEST=1
compiler := 1
useCuda := 0

include ../../Makefile
