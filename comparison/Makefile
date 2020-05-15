BASE = ../base
CPU_DIR = ../cpu-version
CXXFLAGS = -Wall -I.. -I$(BASE) -I$(CPU_DIR) -g

all : evaluation

evaluation : evaluation.o $(BASE)/generator.o $(BASE)/graph.o $(BASE)/matrix_graph.o $(BASE)/sparse_graph.o $(BASE)/list_graph.o $(CPU_DIR)/cpu_prim.o
	g++ evaluation.o $(BASE)/generator.o $(BASE)/graph.o $(BASE)/matrix_graph.o $(BASE)/sparse_graph.o $(BASE)/list_graph.o $(CPU_DIR)/cpu_prim.o -o evaluation

evaluation.o : evaluation.cpp $(BASE)/graph.hpp $(BASE)/matrix_graph.hpp $(BASE)/sparse_graph.hpp $(BASE)/list_graph.hpp $(BASE)/generator.hpp $(CPU_DIR)/cpu_prim.hpp
