BASE = ../../base
CXXFLAGS = -Wall -I.. -I$(BASE) -I$(BASE)/test -DWITH_BOOST

test : test_cpu_prim

test_cpu_prim : test_cpu_prim.o ../cpu_prim.o $(BASE)/graph.o $(BASE)/matrix_graph.o $(BASE)/sparse_graph.o
	g++ $^ -o $@
	./test_cpu_prim || (rm test_cpu_prim; exit 1)

test_cpu_prim.o : test_cpu_prim.cpp ../cpu_prim.hpp $(BASE)/graph.hpp $(BASE)/matrix_graph.hpp $(BASE)/sparse_graph.o

clean :
	rm -f *.o test_cpu_prim
