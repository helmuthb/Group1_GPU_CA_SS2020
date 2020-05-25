BASE = ../base
CPU_DIR = ../cpu-version
THRUST_DIR = ../thrust
CXXFLAGS = -Wall -I.. -I$(BASE) -I$(CPU_DIR) -I$(THRUST_DIR) -g -DWITH_BOOST
NVCC = nvcc
CUDAFLAGS = -O3 -I$(BASE) -I$(CPU_DIR) -I$(THRUST_DIR) -DWITH_BOOST

all : evaluation

evaluation : evaluation.o $(BASE)/generator.o $(BASE)/graph.o $(BASE)/matrix_graph.o $(BASE)/sparse_graph.o $(BASE)/list_graph.o $(CPU_DIR)/cpu_prim.o $(THRUST_DIR)/thrust_prim.o
	$(NVCC) $(CUDAFLAGS) $^ -o $@

evaluation.o : evaluation.cu $(BASE)/graph.hpp $(BASE)/matrix_graph.hpp $(BASE)/sparse_graph.hpp $(BASE)/list_graph.hpp $(BASE)/generator.hpp $(CPU_DIR)/cpu_prim.hpp

%.o : %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean :
	rm -f *.o evaluation
