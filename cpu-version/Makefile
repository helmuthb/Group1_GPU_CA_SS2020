BASE = ../base
CXXFLAGS = -Wall -I$(BASE) -g -DWITH_BOOST

all : cpu_prim check

cpu_prim : cpu_prim_main.o cpu_prim.o cpu_prim2.o $(BASE)/graph.o $(BASE)/matrix_graph.o
	g++ $^ -o $@

check : *.cpp *.hpp
	@cd test && make test

clean :
	rm -f *.o cpu_prim check
	@make -C test clean
