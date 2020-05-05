CXXFLAGS = -Wall

all : generator check

generator : generator_main.o generator.o graph.o
	g++ generator_main.o generator.o graph.o -o generator

graph.o : graph.cpp graph.hpp

generator.o : generator.cpp graph.hpp

generator_main.o : generator.hpp graph.hpp

check : *.cpp
	@cd test && make test
