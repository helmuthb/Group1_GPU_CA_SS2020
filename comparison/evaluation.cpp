#include "graph.hpp"
#include "matrix_graph.hpp"
#include "sparse_graph.hpp"
#include "list_graph.hpp"
#include "generator.hpp"
#include "cpu_prim.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono;

template <class T_GRAPH>
double cpuRuntime(const Graph& g, int cntRuns) {
    steady_clock::time_point begin, end;
    double runtime;

    // allow for warm-up
    T_GRAPH mst;
    cpuPrimAlgorithm(g, mst);
    // now the real test run
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; i++) {
        MatrixGraph mst2;
        // find MST solution
        cpuPrimAlgorithm(g, mst2);
    }
    end = steady_clock::now();
    runtime = (duration_cast<duration<double>>(end-begin)).count();
    // return as miliseconds per round
    return 1000.*runtime/cntRuns;
}

void runParamSet(std::ostream& os, int num_vertices, int weight_range, float density, int numReplica, int cntRuns) {
    for (int i=0; i<numReplica; i++) {
        // create an undirected graph
        MatrixGraph g1;
        generator(g1, num_vertices, 0, weight_range, density, false);
        // run through all implementations and get runtime
        double runtime;
        runtime = cpuRuntime<MatrixGraph>(g1, cntRuns);
        // output to file 
        os << "cpu_m," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
        // create an undirected graph
        SparseGraph g2;
        generator(g2, num_vertices, 0, weight_range, density, false);
        // run through all implementations and get runtime
        runtime = cpuRuntime<SparseGraph>(g2, cntRuns);
        // output to file 
        os << "cpu_s," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
        // create an undirected graph
        ListGraph g3;
        generator(g3, num_vertices, 0, weight_range, density, false);
        // run through all implementations and get runtime
        runtime = cpuRuntime<ListGraph>(g3, cntRuns);
        // output to file 
        os << "cpu_l," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // runParamSet(std::cout, 10000, 50, 0.01, 3, 1);
    runParamSet(std::cout, 1000, 50, 0.1, 10, 10);
    runParamSet(std::cout, 1000, 50, 0.2, 10, 10);
    runParamSet(std::cout, 1000, 50, 0.5, 10, 10);
    runParamSet(std::cout, 1000, 50, 0.75, 10, 10);
    runParamSet(std::cout, 500, 50, 0.2, 10, 20);
    runParamSet(std::cout, 100, 50, 0.2, 10, 100);
}