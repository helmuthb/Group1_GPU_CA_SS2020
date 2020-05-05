BASE = ../base
CXXFLAGS = -Wall -I$(BASE)

all : cpu_prim check

cpu_prim : cpu_prim_main.o cpu_prim.o $(BASE)/graph.o
	g++ cpu_prim_main.o cpu_prim.o $(BASE)/graph.o -o cpu_prim

check : *.cpp
	@cd test && make test
	
