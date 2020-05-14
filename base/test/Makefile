CXXFLAGS = -Wall -I.. -g

test : test_graph test_generator

test_graph : test_graph.o ../graph.o ../matrix_graph.o ../sparse_graph.o
	g++ test_graph.o ../graph.o ../matrix_graph.o ../sparse_graph.o -o test_graph
	./test_graph || (rm test_graph; exit 1)

test_graph.o : test_graph.cpp ../graph.hpp ../matrix_graph.hpp ../sparse_graph.hpp

test_generator : test_generator.o ../generator.o ../graph.o ../matrix_graph.o ../sparse_graph.o
	g++ test_generator.o ../graph.o ../generator.o ../matrix_graph.o ../sparse_graph.o -o test_generator
	./test_generator || (rm test_generator; exit 1)

test_generator.o : test_generator.cpp ../graph.hpp ../matrix_graph.hpp ../sparse_graph.hpp ../generator.hpp
