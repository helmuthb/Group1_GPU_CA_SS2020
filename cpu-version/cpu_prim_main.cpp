#include "cpu_prim.hpp"
#include "graph.hpp"
#include "matrix_graph.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    // read graph from stdin
    MatrixGraph g;
    std::cin >> g;
    // calculate mst
    MatrixGraph mst;
    cpuPrimAlgorithm(g, mst);
    // output MST
    std::cout << mst;
}