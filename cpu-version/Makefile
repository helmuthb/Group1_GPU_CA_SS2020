BASE = ../base
CXXFLAGS = -Wall -I$(BASE) -g -DWITH_BOOST

all : cpu_prim check

cpu_prim : cpu_prim_main.o cpu_prim.o $(BASE)/graph.o $(BASE)/matrix_graph.o
	g++ cpu_prim_main.o cpu_prim.o $(BASE)/graph.o $(BASE)/matrix_graph.o -o cpu_prim

check : *.cpp *.hpp
	@cd test && make test
	
