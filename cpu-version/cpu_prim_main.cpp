#include "cpu_prim.hpp"
#include "graph.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    // read graph from stdin
    Graph g;
    std::cin >> g;
    // calculate mst
    Graph mst = cpuPrimAlgorithm(g);
    // output MST
    std::cout << mst;
}