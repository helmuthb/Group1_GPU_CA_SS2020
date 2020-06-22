BASE = ../base
CXXFLAGS = -Wall -I$(BASE) -DWITH_BOOST

NVCC = nvcc
CUDAFLAGS = -O3 -I$(BASE) -DWITH_BOOST

all : thrust_prim.o

thrust_prim : thrust_prim.o thrust_prim_main.o $(BASE)/graph.o
	$(NVCC) $(CUDAFLAGS) $^ -o $@

thrust_prim.o : thrust_prim.cu thrust_prim.hpp $(BASE)/graph.hpp
	$(NVCC) $(CUDAFLAGS) -c thrust_prim.cu -o thrust_prim.o

check : *.cu *.hpp
	@cd test && make test

clean :
	rm -f *.o thrust_prim
