BASE = ../base
CPU_DIR = ../cpu-version
CXXFLAGS = -Wall -I.. -I$(BASE) -I$(CPU_DIR)

all : evaluation

evaluation : evaluation.o $(BASE)/generator.o $(BASE)/graph.o $(CPU_DIR)/cpu_prim.o
	g++ evaluation.o $(BASE)/generator.o $(BASE)/graph.o $(CPU_DIR)/cpu_prim.o -o evaluation

evaluation.o : evaluation.cpp $(BASE)/graph.hpp $(BASE)/generator.hpp $(CPU_DIR)/cpu_prim.hpp
