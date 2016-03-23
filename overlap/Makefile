#################
#
# Makefile for "Kernels for computation and communication overlap"
#
# Adjust to following for your local setup:

# set path to cuda runtime library for the Fortran linker
LIB_PATH = -L/cm/shared/apps/cuda55/toolkit/current/lib64/

# select the right compute capability for your GPU
COMPUTE_CAPABILITY = 52

#################

NVCCFLAGS = -gencode arch=compute_$(COMPUTE_CAPABILITY),code=sm_$(COMPUTE_CAPABILITY) -Xptxas=-v

COMPARE = $(shell echo $(COMPUTE_CAPABILITY)\>=35 | bc)
ifeq ($(COMPARE), 1)
NVCCFLAGS += -D USE_READ_ONLY_CACHE=1
endif

# Compilers
CC = g++
NVCC = nvcc -Xcompiler=-Wall

OPT = -O3
#OPT = -g

CFLAGS = $(OPT) 

.PHONY: clean

all: conv 

clean:
	rm -f *.o conv 

conv: conv.cu timer.o 
	$(NVCC) $(OPT) $(NVCCFLAGS) -c conv.cu -o conv.o
	$(NVCC) $(OPT) $(NVCCFLAGS) conv.o timer.o -o conv

timer.o: timer.cc timer.h
	$(CC) $(CFLAGS) -c timer.cc -o timer.o

