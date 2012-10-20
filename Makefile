#    Copyright (C) 2009-2010 the North Carolina State University
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

.SUFFIXES: .hpp .cpp .o .d

UNAME   :=  $(shell uname)
HOSTNAME := $(shell hostname)
UNAME_A :=  $(shell uname -a)

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring esheDARWIN, $(OSUPPER)))

ROOTDIR ?= .

#all:$(TARGET)
PROJECTDIR ?= .

#TOOLS_LIB = $(ROOTDIR)/tools/libmytool.a



ifeq ($(useCuda),1)
	BACKEND_INC_DIR = $(ROOTDIR)/runtime/cuda/include
	BACKEND_INC = $(wildcard $(BACKEND_INC_DIR)/*.h)
	BACKEND_SRC_DIR = $(ROOTDIR)/runtime/cuda/src
	BACKEND_SRC = $(wildcard $(BACKEND_SRC_DIR)/*.cpp)
	RUNTIME_TEMPLATE_DIR = $(ROOTDIR)/runtime/cuda/template
	RUNTIME_TEMPLATE_INC = $(wildcard $(ROOTDIR)/runtime/cuda/template/*.cuh)
	BACKEND_CUDA_SRC = $(wildcard $(BACKEND_SRC_DIR)/*.cu)
	BACKEND_OBJ_DIR = $(ROOTDIR)/obj
	BACKEND_OBJ = $(patsubst %.cpp, $(BACKEND_OBJ_DIR)/%.o, $(notdir $(BACKEND_SRC)))
	BACKEND_CUDA_OBJ = $(patsubst %.cu, $(BACKEND_OBJ_DIR)/%_cu.o, $(notdir $(BACKEND_CUDA_SRC)))
else
endif


COMPILER_SRC_DIR = $(ROOTDIR)/compiler/src
#COMPILER_SRC_DIR += $(ROOTDIR)/compiler/src/gpu
COMPILER_INC_DIR = $(ROOTDIR)/compiler/include
COMPILER_OBJ_DIR = $(ROOTDIR)/compiler/obj
COMPILER_SRC = $(wildcard $(COMPILER_SRC_DIR)/*.cpp)
COMPILER_INC = $(wildcard $(COMPILER_INC_DIR)/*.h) 
COMPILER_OBJ = $(patsubst %.cpp, $(COMPILER_OBJ_DIR)/%.o, $(notdir $(COMPILER_SRC)))


BACKEND_USER_OBJ_DIR = ./obj

#CUDA_INSTALL_PATH := /usr/local/cuda
ifeq ($(HOSTNAME),fermi)
    CUDA_INSTALL_PATH := /usr/local/cuda
else
	CUDA_INSTALL_PATH := /usr/local/cuda
endif

ifeq ($(OSUPPER), DARWIN)
	CUDA_INSTALL_LIB := $(CUDA_INSTALL_PATH)/lib
	SMVERSIONFLAG := -arch sm_12
else
	CUDA_INSTALL_LIB := $(CUDA_INSTALL_PATH)/lib64   #TODO detect 64 bit
	SMVERSIONFLAG := -arch sm_20
endif

CUDPP_INC_DIR := $(CUDA_INSTALL_PATH)/cudpp/include
CC :=g++
#CCDEP=g++
NVCC=nvcc
#nvcc

CUBINDIR := $(PROJECTDIR)/obj
CUBINS += $(patsubst %.cu, $(CUBINDIR)/%.cubin, $(notdir $(CUBINFILES)))
CUOS += $(patsubst %.cu, $(CUBINDIR)/%_cu.o, $(notdir $(CUFILES)))

BACKEND_OBJ = $(patsubst %.cpp, $(BACKEND_OBJ_DIR)/%.o, $(notdir $(BACKEND_SRC)))
BACKEND_CUDA_OBJ = $(patsubst %.cu, $(BACKEND_OBJ_DIR)/%_cu.o, $(notdir $(BACKEND_CUDA_SRC)))


USER_MAIN_OBJ = $(patsubst %.cpp, %.o, $(USER_MAIN_FILES))
USER_OBJ = $(patsubst %.c, %.o, $(USER_FILES))
USER_CUDA_OBJ = $(patsubst %.cu, $(BACKEND_USER_OBJ_DIR)/%_cu.o, $(USER_CUDA_FILES))


ifeq ($(release),1)
	CPPFLAGS += -O3 -Wall -fopenmp
	NVCCFLAG += -O2 
else
    CPPFLAGS +=-g -D_DEBUG  -fopenmp -Wall #-pg -Wall 
	NVCCFLAG +=-g -D_DEBUG -G -lpthread 
endif

CPPFLAGS +=-I$(CUDPP_INC_DIR) -I$(CUDA_INSTALL_PATH)/include -L$(CUDA_INSTALL_LIB) -I$(ROOTDIR)/include -I$(ROOTDIR)/tools -I. -I$(ROOTDIR)/tools/boost_1_47_0  -I$(BACKEND_INC_DIR)
NVCCFLAG +=-I. -I$(ROOTDIR)/runtime/cuda -I$(ROOTDIR)/runtime/cuda/template -I$(CUDPP_INC_DIR) -Xptxas -v --compiler-options -fno-strict-aliasing -I$(ROOTDIR)/include -I$(ROOTDIR)/tools/boost_1_47_0 -I$(BACKEND_INC_DIR)  #-I$(ROOTDIR)/tools/cudpp_2.0/include

LIB = -lpthread -lcudart -lcublas -lcurand #-lcudpp 


NVCCFLAG += $(SMVERSIONFLAG)

ifeq ($(UNIT_TEST),1)
   GTEST =$(ROOTDIR)/unittest/gtest-1.6.0/libgtest.a
   CPPFLAGS += -I$(ROOTDIR)/unittest/gtest-1.6.0/include 
  NVCCFLAG +=-I$(ROOTDIR)/unittest/gtest-1.6.0/include  
endif

ifeq ($(USECURAND),1)
   LIB += -lcurand 
endif

#-Wall  # enable later

# enable terminal output or not
#VERBOSE = @  
AR = ar

$(BACKEND_USER_OBJ_DIR)/%_cu.o:  %.cu $(BACKEND_INC) $(USER_H_FILES) $(RUNTIME_TEMPLATE_INC)
	mkdir -p ./obj
	$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAG) -c -o $@  -m64 $<

$(BACKEND_OBJ_DIR)/%_cu.o: $(BACKEND_SRC_DIR)/%.cu
	$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAG) -c -o $@  -m64 $<

$(BACKEND_OBJ_DIR)/%.o: $(BACKEND_SRC_DIR)/%.cpp 
	mkdir -p $(BACKEND_OBJ_DIR)
	$(VERBOSE)$(CC) $(CPPFLAGS) -I$(ROOTDIR)/include -o $@ -c $<

$(COMPILER_OBJ_DIR)/%.o: $(COMPILER_SRC_DIR)/%.cpp $(COMPILER_INC)
	mkdir -p $(COMPILER_OBJ_DIR)
	$(VERBOSE)$(CC) $(CPPFLAGS) -I$(ROOTDIR)/compiler/include -c -o $@  $<

%.o: %.cpp $(BACKEND_INC)
	$(VERBOSE)$(CC) $(CPPFLAGS) -I$(ROOTDIR)/include -o $@ -c $<

#$(CUNESL_USER_OBJ_DIR)/%.o: %.cpp $(USER_H_FILES)
#	mkdir -p ./obj
#	$(VERBOSE)$(CC) $(CPPFLAGS) -I$(ROOTDIR)/include -o $@ -c $<

#$(CUNESL_USER_OBJ_DIR)/%.o: %.c
#	mkdir -p ./obj
#	$(VERBOSE)$(CC) $(CPPFLAGS) -I$(ROOTDIR)/include -o $@ -c $<


ifeq ($(compiler),1)
$(TARGET): Makefile $(USER_OBJ) $(COMPILER_OBJ) $(USER_CUDA_OBJ) $(LEX_YY_O) $(YACC_TAB_O) $(LEX_YY_C) $(USER_MAIN_FILE)
	mkdir -p $(ROOTDIR)/bin
	$(CC) $(CPPFLAGS) $(USER_OBJ) $(COMPILER_OBJ) $(USER_MAIN_FILE) -o $@ 
else
$(TARGET): Makefile $(USER_OBJ) $(BACKEND_OBJ) $(BACKEND_CUDA_OBJ) $(USER_CUDA_OBJ) $(LEX_YY_O) $(YACC_TAB_O) $(LEX_YY_C) $(USER_MAIN_FILE) 
	mkdir -p $(ROOTDIR)/bin
	$(CC) $(CPPFLAGS) $(CUDA_INCLUDE) $(USER_OBJ) $(BACKEND_OBJ) $(USER_CUDA_OBJ) $(BACKEND_CUDA_OBJ) $(USER_MAIN_FILE) -o $@ $(GTEST) $(LIB)
endif


cubindirectory:
	mkdir -p $(CUBINDIR)

#gstreamlibobjdirectory:
#	mkdir -p $(GSTREAMLIB_OBJ_DIR)

gstreamlibdirectory:
	mkdir -p $(ROOTDIR)/lib

clean:
	rm -rf $(TARGET)  $(COMPILER_OBJ_DIR)/*.o $(CUNESL_OBJ_DIR)/*.o $(CUNESL_OBJ_DIR)/*.o $(USER_CUDA_OBJ) ./*.o
	rm -rf $(ROOTDIR)/obj/*.o
	rm -rf $(YACC_TAB_C) $(YACC_TAB_H) $(LEX_YY_C) $(LEX_YY_O)

