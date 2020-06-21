BASE = ../../base
CXXFLAGS = -Wall -I.. -I$(BASE) -I$(BASE)/test -DWITH_BOOST

NVCC = nvcc
CUDAFLAGS = -O3 -I.. -I$(BASE) -I$(BASE)/test -DWITH_BOOST

test : test_thrust_prim

test_thrust_prim : test_thrust_prim.o ../thrust_prim.o $(BASE)/graph.o $(BASE)/list_graph.o $(BASE)/generator.o
	$(NVCC) $(CUDAFLAGS) $^ -o $@
	./test_thrust_prim || (rm test_thrust_prim; exit 1)

test_thrust_prim.o : test_thrust_prim.cu ../thrust_prim.hpp $(BASE)/graph.hpp $(BASE)/list_graph.hpp
	$(NVCC) $(CUDAFLAGS) -c test_thrust_prim.cu -o $@

clean :
	rm -f *.o test_thrust_prim
