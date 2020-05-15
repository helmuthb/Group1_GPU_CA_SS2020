#include "graph.hpp"
#include "matrix_graph.hpp"
#include "sparse_graph.hpp"
#include "list_graph.hpp"
#include "thrust_prim.hpp"
#include "generator.hpp"
#include "cpu_prim.hpp"
#include <chrono>
#include <iostream>
#ifdef WITH_BOOST
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#endif

using namespace std::chrono;

double thrustRuntime(const Graph& g, int cntRuns) {
    steady_clock::time_point begin, end;
    double runtime;

    // prepare data for thrust
    thrust::host_vector<uint32_t> num_edges;
    thrust::host_vector<uint32_t> idx_edges;
    thrust::host_vector<uint32_t> target;
    thrust::host_vector<int32_t> weight;
    thrustPrepare(g, &num_edges, &idx_edges, &target, &weight);
    thrust::host_vector<uint32_t> predecessor;
    // allow for warm-up
    thrustPrimAlgorithm(&num_edges, &idx_edges, &target, &weight, &predecessor);
    // now the real test run
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; ++i) {
        // find MST solution
        thrustPrimAlgorithm(&num_edges, &idx_edges, &target, &weight, &predecessor);
    }
    end = steady_clock::now();
    runtime = (duration_cast<duration<double>>(end-begin)).count();
    // return as miliseconds per round
    return 1000.*runtime/cntRuns;    
}

template <class T_GRAPH>
double cpuRuntime(const Graph& g, int cntRuns) {
    steady_clock::time_point begin, end;
    double runtime;

    // allow for warm-up
    T_GRAPH mst;
    cpuPrimAlgorithm(g, mst);
    // now the real test run
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; ++i) {
        MatrixGraph mst2;
        // find MST solution
        cpuPrimAlgorithm(g, mst2);
    }
    end = steady_clock::now();
    runtime = (duration_cast<duration<double>>(end-begin)).count();
    // return as miliseconds per round
    return 1000.*runtime/cntRuns;
}

#ifdef WITH_BOOST
struct do_nothing_dijkstra_visitor : boost::default_dijkstra_visitor {};

double boostRuntime(const Graph& g, int cntRuns) {
    steady_clock::time_point begin, end;
    BoostGraph boost_g;
    double runtime;

    // allow for warm-up
    g.toBoost(boost_g);
    auto p = std::vector<boost::graph_traits<BoostGraph>::vertex_descriptor >(g.num_vertices());
    boost::prim_minimum_spanning_tree(boost_g, &p[0]);
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; ++i) {
        boost::prim_minimum_spanning_tree(boost_g, &p[0]);
    }
    end = steady_clock::now();
    runtime = (duration_cast<duration<double>>(end-begin)).count();
    // return as miliseconds per round
    return 1000.*runtime/cntRuns;
}
#endif

void runParamSet(std::ostream& os, int num_vertices, int weight_range, float density, int numReplica, int cntRuns) {
    for (int i=0; i<numReplica; ++i) {
        // create an undirected graph
        MatrixGraph g;
        generator(g, num_vertices, 0, weight_range, density, false);
        // run through all implementations and get runtime
        double runtime;
        runtime = cpuRuntime<MatrixGraph>(g, cntRuns);
        // output to file 
        os << "cpu_m," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
        // create an undirected graph
        SparseGraph g2 = g;
        // run through all implementations and get runtime
        runtime = cpuRuntime<SparseGraph>(g2, cntRuns);
        // output to file 
        os << "cpu_s," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
        // create an undirected graph
        ListGraph g3 = g;
        // run through all implementations and get runtime
        runtime = cpuRuntime<ListGraph>(g3, cntRuns);
        // output to file 
        os << "cpu_l," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
#ifdef WITH_BOOST
        // run through boost implementation
        runtime = boostRuntime(g, cntRuns);
        // output to file 
        os << "cpu_b," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
#endif
        // run through thrust implementation
        runtime = thrustRuntime(g, cntRuns);
        // output to file 
        os << "thrust," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
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