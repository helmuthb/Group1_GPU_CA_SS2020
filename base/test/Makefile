CXXFLAGS = -Wall -I..

test : test_graph test_generator

test_graph : test_graph.o ../graph.o
	g++ test_graph.o ../graph.o -o test_graph
	./test_graph || (rm test_graph; exit 1)

test_generator : test_generator.o ../generator.o ../graph.o
	g++ test_generator.o ../graph.o ../generator.o -o test_generator
	./test_generator || (rm test_generator; exit 1)
