#include "graph.hpp"
#include "matrix_graph.hpp"
#include "sparse_graph.hpp"
#include "list_graph.hpp"
#include "thrust_prim.hpp"
#include "cuda1_prim.hpp"
#include "cuda2_prim.hpp"
#include "generator.hpp"
#include "cpu_prim.hpp"
#include <chrono>
#include <iostream>
#ifdef WITH_BOOST
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#endif

using namespace std::chrono;

double cuda1Runtime(const Graph& g, int cntRuns, Graph& mst) {
    steady_clock::time_point begin, end;
    double runtime;

    const uint32_t V = g.num_vertices();
    // Each edge is present twice: once from each vertex
    const uint32_t E = g.num_edges();

    // Inputs
    uint2 *outbound_vertices = new uint2[V];
    uint2 *inbound_vertices = new uint2[E*2];
    // Outputs
    uint32_t *outbound = new uint32_t[V];
    uint32_t *inbound = new uint32_t[V];
    uint32_t *weights = new uint32_t[V];

    // Prepare input data
    cuda1Setup(g, inbound_vertices, outbound_vertices);

    // initialize solution arrays with +inf
    std::fill(outbound, outbound + V, UINT32_MAX);
    std::fill(inbound, inbound + V, UINT32_MAX);
    std::fill(weights, weights + V, UINT32_MAX);

    // allow for warm-up
    cuda1PrimAlgorithm(V, E, outbound_vertices, inbound_vertices, outbound, inbound, weights);

    // now the real test run
    begin = steady_clock::now();
    for (int i = 0; i < cntRuns; ++i) {
        // initialize solution arrays with +inf
        std::fill(outbound, outbound + V, UINT32_MAX);
        std::fill(inbound, inbound + V, UINT32_MAX);
        std::fill(weights, weights + V, UINT32_MAX);
        // find MST solution
        cuda1PrimAlgorithm(V, E, outbound_vertices, inbound_vertices, outbound, inbound, weights);
    }
    end = steady_clock::now();
    runtime = (duration_cast<duration<double>>(end - begin)).count();

    mst.resize(V, V - 1, g.is_directed());
    // remove invalid edge
    for (uint32_t i = 0; i < V; ++i) {
        if ((uint32_t)inbound[i] <= V) {
            mst.set(outbound[i], inbound[i], (uint32_t)weights[i]);
        }
    }
    delete[] outbound_vertices;
    delete[] inbound_vertices;
    delete[] outbound;
    delete[] inbound;
    delete[] weights;

    // return as miliseconds per round
    return 1000.*runtime / cntRuns;
}

double cuda2Runtime(const Graph& g, int cntRuns, Graph& mst,
                    bool pinned=false, bool zerocopy=false) {
    steady_clock::time_point begin, end;
    double runtime;

    const uint32_t V = g.num_vertices();
    // Each edge is present twice: once from each vertex
    const uint32_t E = 2*g.num_edges();

    // Inputs
    uint2 *vertices, *edges;
    // Outputs
    uint32_t *outbound, *inbound, *weights;

    //
    // Allocate the inputs and outputs depending on the memory strategy we want
    // to evaluate:
    //
    //   * pinned == false, zerocopy == false
    //     -> regular memory
    //
    //   * pinned == true, zeropopy == false
    //     -> pin host-allocated data, but do nothing for device-allocated data
    //
    //   * pinned == true, zerocopy == true
    //     -> Allocate everything on the host, use device pointers
    //
    if (!pinned) {
        vertices = new uint2[V];
        edges = new uint2[E];
        outbound = new uint32_t[V-1];
        inbound = new uint32_t[V-1];
        weights = new uint32_t[V-1];
    } else {
        if (!zerocopy) { 
            cudaMallocHost((uint2 **)    &vertices, V     * sizeof(uint2));
            cudaMallocHost((uint2 **)    &edges,    E     * sizeof(uint2));
            cudaMallocHost((uint32_t **) &outbound, (V-1) * sizeof(uint32_t));
            cudaMallocHost((uint32_t **) &inbound,  (V-1) * sizeof(uint32_t));
            cudaMallocHost((uint32_t **) &weights,  (V-1) * sizeof(uint32_t));
        } else {
            cudaHostAlloc((uint2 **)    &vertices, V     * sizeof(uint2),    cudaHostAllocMapped);
            cudaHostAlloc((uint2 **)    &edges,    E     * sizeof(uint2),    cudaHostAllocMapped);
            cudaHostAlloc((uint32_t **) &outbound, (V-1) * sizeof(uint32_t), cudaHostAllocMapped);
            cudaHostAlloc((uint32_t **) &inbound,  (V-1) * sizeof(uint32_t), cudaHostAllocMapped);
            cudaHostAlloc((uint32_t **) &weights,  (V-1) * sizeof(uint32_t), cudaHostAllocMapped);
        }
    }

    // Prepare input data
    cuda2Setup(g, vertices, edges);

    // allow for warm-up
    cuda2PrimAlgorithm(vertices, V, edges, E, outbound, inbound, weights, zerocopy);

    // now the real test run
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; ++i) {
        // find MST solution
        cuda2PrimAlgorithm(vertices, V, edges, E, outbound, inbound, weights, zerocopy);
    }
    end = steady_clock::now();
    runtime = (duration_cast<duration<double>>(end-begin)).count();

    mst.resize(V, V-1, g.is_directed());
    for (uint32_t i = 0; i < V-1; ++i) {
        mst.set(outbound[i], inbound[i], (uint32_t) weights[i]);
    }

    if (!pinned) {
        delete[] vertices;
        delete[] edges;
        delete[] outbound;
        delete[] inbound;
        delete[] weights;
    } else {
        cudaFreeHost(vertices);
        cudaFreeHost(edges);
        cudaFreeHost(outbound);
        cudaFreeHost(inbound);
        cudaFreeHost(weights);
    }

    // return as miliseconds per round
    return 1000.*runtime/cntRuns;    
}

double thrustRuntime(const Graph& g, int cntRuns, Graph& mst) {
    steady_clock::time_point begin, end;
    double runtime;

    // prepare data for thrust
    const uint32_t V = g.num_vertices();
    // Each edge is present twice: once from each vertex
    const uint32_t E = g.num_edges();

    thrust::host_vector<uint2> vertices(V);
    thrust::host_vector<uint2> edges(2*E);
    thrustSetup(g, vertices, edges);
    thrust::host_vector<uint32_t> mst_in(V);
    thrust::host_vector<uint32_t> mst_out(V);
    thrust::host_vector<uint32_t> mst_weight(V);
    // allow for warm-up
    thrustPrimAlgorithm(vertices, edges, mst_out, mst_in, mst_weight, V, E);
    // now the real test run
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; ++i) {
        // find MST solution
        thrustPrimAlgorithm(vertices, edges, mst_out, mst_in, mst_weight, V, E);
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
        MatrixGraph g;
        uint64_t itseed = seed+i;
        generator(g, num_vertices, 0, weight_range, density, false, itseed);
        // run through all implementations and get runtime
        double runtime;

        /*ListGraph cpu_l_mst;
        runtime = cpuRuntime<ListGraph>(g, cntRuns, cpu_l_mst);
        // output to file 
        os << "cpu_l," << i
                << "," << itseed
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << cpu_l_mst.sum_weights()
                << std::endl;
	*/
#ifdef WITH_BOOST
        // run through boost implementation
        ListGraph boost_mst;
        runtime = boostRuntime(g, cntRuns, boost_mst);
        // output to file 
        os << "boost," << i
                << "," << itseed
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << boost_mst.sum_weights()
                << std::endl;
#endif
/* */
        // run through thrust implementation
        ListGraph thrust_mst;
        runtime = thrustRuntime(g, cntRuns, thrust_mst);
        // output to file 
        os << "thrust," << i
                << "," << itseed
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << thrust_mst.sum_weights()
                << std::endl;
/* */

        // run through CUDA implementation #1
        ListGraph cuda1_mst;
        runtime = cuda1Runtime(g, cntRuns, cuda1_mst);
        // output to file 
        os << "cuda1," << i
            << "," << itseed
            << "," << num_vertices
            << "," << density
            << "," << weight_range
            << "," << runtime
            << "," << cuda1_mst.sum_weights()
            << std::endl;

        // run through CUDA implementation #2 - regular
        ListGraph cuda2_mst;
        runtime = cuda2Runtime(g, cntRuns, cuda2_mst, false, false);
        // output to file
        os << "cuda2," << i
                << "," << itseed
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << cuda2_mst.sum_weights()
                << std::endl;

        // run through CUDA implementation #2 - pinned memory
        ListGraph cuda2_mst_pinned;
        runtime = cuda2Runtime(g, cntRuns, cuda2_mst_pinned, true, false);
        // output to file
        os << "cuda2-pinned," << i
                << "," << itseed
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << cuda2_mst_pinned.sum_weights()
                << std::endl;

        // run through CUDA implementation #2 - pinned memory
        ListGraph cuda2_mst_zero;
        runtime = cuda2Runtime(g, cntRuns, cuda2_mst_zero, true, true);
        // output to file
        os << "cuda2-zero," << i
                << "," << itseed
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << cuda2_mst_zero.sum_weights()
                << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "implementation,run,seed,vertices,density,weight_range,runtime,min" << std::endl;
    runParamSet(std::cout, 100, 5000, 0.5, 3, 1, 42);
    runParamSet(std::cout, 100, 5000, 0.9, 3, 1, 42);
    runParamSet(std::cout, 512, 5000, 1.0, 3, 1, 42);
    runParamSet(std::cout, 1000, 5000, 0.2, 3, 1, 42);
    runParamSet(std::cout, 1000, 5000, 0.5, 3, 1, 42);
    runParamSet(std::cout, 1000, 5000, 0.7, 3, 1, 42);
    runParamSet(std::cout, 1000, 5000, 0.9, 3, 1, 42);
    runParamSet(std::cout, 4096, 5000, 0.01, 3, 1, 42);
    runParamSet(std::cout, 4097, 5000, 0.01, 3, 1, 42);
    //runParamSet(std::cout, 4096, 5000, 1, 3, 1, 42);
    runParamSet(std::cout, 5000, 5000, 0.001, 3, 1, 42);
    runParamSet(std::cout, 10000, 50000, 0.0001, 3, 1, 42);
    //runParamSet(std::cout, 50000, 50000, 0.00001, 3, 1, 42);
}
