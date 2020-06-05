#include "graph.hpp"
#include "matrix_graph.hpp"
#include "sparse_graph.hpp"
#include "list_graph.hpp"
#include "thrust_prim.hpp"
#include "cuda_prim.hpp"
#include "generator.hpp"
#include "cpu_prim.hpp"
#include <chrono>
#include <iostream>
#ifdef WITH_BOOST
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#endif

using namespace std::chrono;

double cudaRuntime(const Graph& g, int cntRuns, Graph& mst) {
    steady_clock::time_point begin, end;
    double runtime;

    // prepare data for CUDA
    uint2 * inbound_vertices, *outbound_vertices, *shape = NULL;
    cudaSetup(g, inbound_vertices, outbound_vertices, shape);
    const uint32_t V = shape->x;

    uint32_t *outbound = new uint32_t[V];
    uint32_t *inbound = new uint32_t[V];
    uint32_t *weights = new uint32_t[V];

    // allow for warm-up
    cudaPrimAlgorithm(inbound_vertices, outbound_vertices, shape,
        inbound, outbound, weights);

    // now the real test run
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; ++i) {
        // find MST solution
        cudaPrimAlgorithm(inbound_vertices, outbound_vertices, shape,
            inbound, outbound, weights);
        }
    end = steady_clock::now();
    runtime = (duration_cast<duration<double>>(end-begin)).count();

    // store the result
    // FIXME: This is currently deactivated as there seems to be a bug in the
    // CUDA implementation. For example, given 10 nodes, 0.5 density, weight
    // 5000, I get the following values in outbound/inbound/weights:
    //
    //    2261431280 -> 0     w=4294967295
    //    21874 -> 21874     w=4294967295
    //    0 -> 0     w=1510
    //    0 -> 0     w=4294967295
    //    0 -> 7     w=1553
    //    0 -> 0     w=4115
    //    0 -> 4     w=2081
    //    1491 -> 6     w=4294967295
    //    6 -> 1     w=4294967295
    //    4187 -> 2     w=4294967295
    //
    mst.resize(g.num_vertices(), g.num_vertices()-1, g.is_directed());
    for (uint32_t i = 0; i < V; ++i) {
        ;
        // FIXME: CUDA implementation uses unsigned weights
        //mst.set(outbound[i], inbound[i], (uint32_t) weights[i]);
    }

    delete[] inbound_vertices;
    delete[] outbound_vertices;
    delete[] shape;

    delete[] inbound;
    delete[] outbound;
    delete[] weights;

    // return as miliseconds per round
    return 1000.*runtime/cntRuns;    
}

double thrustRuntime(const Graph& g, int cntRuns, Graph& mst) {
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
    // TODO: Store the results in mst
    // mst.resize(g.num_vertices(), g.num_vertices()-1, g.is_directed());
    // ...
    // return as miliseconds per round
    return 1000.*runtime/cntRuns;    
}

template <class T_GRAPH>
double cpuRuntime(const Graph& g, int cntRuns, Graph& mst) {
    steady_clock::time_point begin, end;
    double runtime;

    // allow for warm-up, store the result
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

double boostRuntime(const Graph& g, int cntRuns, Graph& mst) {
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

    // store the result
    mst.resize(g.num_vertices(), g.num_vertices()-1, g.is_directed());
    for (std::size_t i = 0; i != p.size(); ++i) {
        if (p[i] != i) {
            mst.set(i, p[i], g(i, p[i]));
        }
    }
    // return as miliseconds per round
    return 1000.*runtime/cntRuns;
}
#endif

void runParamSet(std::ostream& os, int num_vertices, int weight_range, float density,
                 int numReplica, int cntRuns, uint64_t seed) {
    for (int i=0; i<numReplica; ++i) {
        // create an undirected graph, using a different seed in each replica
        ListGraph g;
        generator(g, num_vertices, 0, weight_range, density, false, seed+numReplica);
        // run through all implementations and get runtime
        double runtime;

        ListGraph cpu_l_mst;
        runtime = cpuRuntime<ListGraph>(g, cntRuns, cpu_l_mst);
        // output to file 
        os << "cpu_l," << i
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << cpu_l_mst.sum_weights()
                << std::endl;
#ifdef WITH_BOOST
        // run through boost implementation
        ListGraph boost_mst;
        runtime = boostRuntime(g, cntRuns, boost_mst);
        // output to file 
        os << "boost," << i
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << boost_mst.sum_weights()
                << std::endl;
#endif
/*
        // run through thrust implementation
        ListGraph thrust_mst;
        runtime = thrustRuntime(g, cntRuns, thrust_mst);
        // output to file 
        os << "thrust," << i
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << thrust_mst.sum_weights()
                << std::endl;
*/
        // run through cuda implementation
        ListGraph cuda_mst;
        runtime = cudaRuntime(g, cntRuns, cuda_mst);
        // output to file 
        os << "cuda," << i
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << -1 // cpu_l_mst.sum_weights()   FIXME, sie cudaRuntime above
                << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "implementation,vertices,density,weight_range,runtime,min" << std::endl;
    runParamSet(std::cout, 10000, 50,  0.01, 3, 1, 42);
    runParamSet(std::cout, 50000, 50, 0.001, 3, 1, 42);
}
