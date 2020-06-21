BASE = ../base
CPU_DIR = ../cpu-version
THRUST_DIR = ../thrust
CUDA1_DIR = ../cuda1
CUDA_DIR = ../cuda
CXXFLAGS = -Wall -I.. -I$(BASE) -I$(CPU_DIR) -I$(THRUST_DIR) -I$(CUDA1_DIR)-I$(CUDA_DIR)  -g -DWITH_BOOST
NVCC = nvcc
CUDAFLAGS = -O3 -I$(BASE) -I$(CPU_DIR) -I$(THRUST_DIR) -I$(CUDA1_DIR) -I$(CUDA_DIR) -DWITH_BOOST

all : evaluation

evaluation : evaluation.o $(BASE)/generator.o $(BASE)/graph.o $(BASE)/matrix_graph.o $(BASE)/sparse_graph.o $(BASE)/list_graph.o $(CPU_DIR)/cpu_prim.o $(THRUST_DIR)/thrust_prim.o $(CUDA1_DIR)/cuda1_prim.o $(CUDA_DIR)/cuda_prim.o
	$(NVCC) $(CUDAFLAGS) $^ -o $@

evaluation.o : evaluation.cu $(BASE)/graph.hpp $(BASE)/matrix_graph.hpp $(BASE)/sparse_graph.hpp $(BASE)/list_graph.hpp $(BASE)/generator.hpp $(CPU_DIR)/cpu_prim.hpp $(THRUST_DIR)/thrust_prim.hpp $(CUDA1_DIR)/cuda1_prim.hpp $(CUDA_DIR)/cuda_prim.hpp

%.o : %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean :
	rm -f *.o evaluation
