BASE = ../base
CXXFLAGS = -Wall -I$(BASE) -DWITH_BOOST

NVCC = nvcc
CUDAFLAGS = -O3 -I$(BASE) -DWITH_BOOST

all : cuda1_prim.o

%.o : %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean :
	rm -f *.o
