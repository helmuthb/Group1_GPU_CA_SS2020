CXXFLAGS = -Wall -g -DWITH_BOOST

all : generator check

generator : generator_main.o generator.o graph.o matrix_graph.o
	g++ $^ -o $@

graph.o : graph.cpp graph.hpp

matrix_graph.o : matrix_graph.cpp matrix_graph.hpp graph.hpp

sparse_graph.o : sparse_graph.cpp sparse_graph.hpp graph.hpp

list_graph.o : list_graph.cpp list_graph.hpp graph.hpp

generator.o : generator.cpp graph.hpp

generator_main.o : generator.hpp graph.hpp matrix_graph.hpp

check : *.cpp *.hpp graph.o matrix_graph.o sparse_graph.o list_graph.o generator.o
	@cd test && make test

clean :
	rm -f *.o generator
	@make -C test clean
