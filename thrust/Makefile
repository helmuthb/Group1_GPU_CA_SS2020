BASE = ../base
CXXFLAGS = -Wall -I$(BASE) -DWITH_BOOST

NVCC = nvcc
CUDAFLAGS = -O3 -I$(BASE) -DWITH_BOOST

all : thrust_prim.o

thrust_prim : thrust_prim.o thrust_prim_main.o $(BASE)/graph.o
	$(NVCC) $(CUDAFLAGS) $^ -o $@

%.o : %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean :
	rm -f *.o thrust_prim
