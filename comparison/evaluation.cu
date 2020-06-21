#include "graph.hpp"
#include "matrix_graph.hpp"
#include "sparse_graph.hpp"
#include "list_graph.hpp"
#include "thrust_prim.hpp"
#include "cuda_prim.hpp"
#include "generator.hpp"
#include "cpu_prim.hpp"
#include "cpu_prim2.hpp"
#include <chrono>
#include <iostream>
#ifdef WITH_BOOST
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#endif

using namespace std::chrono;

double cudaRuntime(const Graph& g, int cntRuns) {
    steady_clock::time_point begin, end;
    double runtime;

    // prepare data for CUDA
    uint2 * vertex_adjacent_count_index, *edge_target_weight;
    uint32_t V = g.num_vertices();
    uint32_t E = g.num_edges();

    vertex_adjacent_count_index = new uint2[V];
    edge_target_weight = new uint2[2*E];
    cudaSetup(g, vertex_adjacent_count_index, edge_target_weight);

    uint32_t *mst_out = new uint32_t[V];
    uint32_t *mst_in = new uint32_t[V];
    uint32_t *mst_weight = new uint32_t[V];

    // allow for warm-up
    cudaPrimAlgorithm(vertex_adjacent_count_index, edge_target_weight,
        mst_out, mst_in, mst_weight, V, E);

    // now the real test run
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; ++i) {
        // find MST solution
        cudaPrimAlgorithm(vertex_adjacent_count_index, edge_target_weight,
            mst_out, mst_in, mst_weight, V, E);
    }
    end = steady_clock::now();
    runtime = (duration_cast<duration<double>>(end-begin)).count();

    delete[] vertex_adjacent_count_index;
    delete[] edge_target_weight;

    delete[] mst_in;
    delete[] mst_out;
    delete[] mst_weight;

    // return as miliseconds per round
    return 1000.*runtime/cntRuns;    
}

double thrustRuntime(const Graph& g, int cntRuns) {
    steady_clock::time_point begin, end;
    double runtime;

    uint32_t V = g.num_vertices();
    uint32_t E = g.num_edges();
    // prepare data for thrust
    thrust::host_vector<uint2> vertex_adjacent_count_index(V);
    thrust::host_vector<uint2> edge_target_weight(2*E);
    thrustSetup(g, vertex_adjacent_count_index, edge_target_weight);
    thrust::host_vector<uint32_t> mst_out(V);
    thrust::host_vector<uint32_t> mst_in(V);
    thrust::host_vector<uint32_t> mst_weight(V);
    // allow for warm-up
    thrustPrimAlgorithm(vertex_adjacent_count_index, edge_target_weight,
        mst_out, mst_in, mst_weight, V, E);
    // now the real test run
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; ++i) {
        // find MST solution
        thrustPrimAlgorithm(vertex_adjacent_count_index, edge_target_weight,
            mst_out, mst_in, mst_weight, V, E);
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

template <class T_GRAPH>
double cpuRuntime2(const Graph& g, int cntRuns) {
    steady_clock::time_point begin, end;
    double runtime;

    // allow for warm-up
    T_GRAPH mst;
    cpuPrim2Algorithm(g, mst);
    // now the real test run
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; ++i) {
        MatrixGraph mst2;
        // find MST solution
        cpuPrim2Algorithm(g, mst2);
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
        ListGraph g;
        generator(g, num_vertices, 0, weight_range, density, false);
        // run through all implementations and get runtime
        double runtime;
/*
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
        // ListGraph g3 = g;
        // run through all implementations and get runtime
*/
        runtime = cpuRuntime<ListGraph>(g, cntRuns);
        // output to file 
        os << "cpu_l," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
/*
        runtime = cpuRuntime2<ListGraph>(g, cntRuns);
        // output to file 
        os << "cpu2_l," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
*/
/*
#ifdef WITH_BOOST
        // run through boost implementation
        runtime = boostRuntime(g, cntRuns);
        // output to file 
        os << "cpu_b," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
#endif
*/
/* */
        // run through thrust implementation
        runtime = thrustRuntime(g, cntRuns);
        // output to file 
        os << "thrust," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
/* */
        // run through cuda implementation
        runtime = cudaRuntime(g, cntRuns);
        // output to file 
        os << "cuda," << i << "," << num_vertices << "," << density << "," << weight_range << "," << runtime << std::endl;
    }
}

int main(int argc, char* argv[]) {
    runParamSet(std::cout, 1000, 50, 0.01, 3, 1);
    runParamSet(std::cout, 5000, 50, 0.001, 3, 1);
    /*
    runParamSet(std::cout, 1000, 50, 0.1, 10, 10);
    runParamSet(std::cout, 1000, 50, 0.2, 10, 10);
    runParamSet(std::cout, 1000, 50, 0.5, 10, 10);
    runParamSet(std::cout, 1000, 50, 0.75, 10, 10);
    runParamSet(std::cout, 500, 50, 0.2, 10, 20);
    runParamSet(std::cout, 100, 50, 0.2, 10, 100);
    */
}