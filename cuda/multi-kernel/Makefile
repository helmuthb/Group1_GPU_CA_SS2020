BASE = ../../base
CXXFLAGS = -Wall -I$(BASE) -DWITH_BOOST

NVCC = nvcc
CUDAFLAGS = -O3 -I$(BASE) -DWITH_BOOST

all : cuda_multi_prim

cuda_multi_prim : cuda_multi_main.o $(BASE)/matrix_graph.o $(BASE)/graph.o $(BASE)/generator.o
	$(NVCC) $(CUDAFLAGS) $^ -o $@

%.o : %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean :
	rm -f *.o cuda_multi_prim
