ROOTDIR?=../..

TARGET := spmv_h


USER_MAIN_FILE := main.cpp
USER_CUDA_FILES := spmv.cu ell.cu coo.cu spmv_kernel.cu ell_kernel.cu coo_kernel.cu
USER_H_FILES := kernel.h 
#USER_FILES := $(YACC_TAB_C) $(LEX_YY_C) 
#USER_H_FILES := $(YACC_TAB_H)
compiler := 0
useCuda := 1

include ../../Makefile

