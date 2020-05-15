BASE = ../base
CXXFLAGS = -Wall -I$(BASE) -DWITH_BOOST

NVCC = nvcc
CUDAFLAGS = -O3 -I$(BASE) -DWITH_BOOST

all : thrust_prim.o

thrust_prim : thrust_prim.o thrust_prim_main.o $(BASE)/graph.o
	$(NVCC) $(CUDAFLAGS) thrust_prim.o thrust_prim_main.o $(BASE)/graph.o -o thrust_prim

%.o : %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@
