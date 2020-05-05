BASE = ../../base
CXXFLAGS = -Wall -I.. -I$(BASE) -I$(BASE)/test

test : test_cpu_prim

test_cpu_prim : test_cpu_prim.o ../cpu_prim.o $(BASE)/graph.o
	g++ test_cpu_prim.o ../cpu_prim.o $(BASE)/graph.o -o test_cpu_prim
	./test_cpu_prim || (rm test_cpu_prim; exit 1)
