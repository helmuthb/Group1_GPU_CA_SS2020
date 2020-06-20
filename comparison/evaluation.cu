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

    const uint32_t V = g.num_vertices();
    // Each edge is present twice: once from each vertex
    const uint32_t E = 2*g.num_edges();

    // Inputs
    uint2 *vertices = new uint2[V];
    uint2 *edges = new uint2[E];
    // Outputs
    uint32_t *outbound = new uint32_t[V-1];
    uint32_t *inbound = new uint32_t[V-1];
    uint32_t *weights = new uint32_t[V-1];

    // Prepare input data
    cudaSetup(g, vertices, edges);

    // allow for warm-up
    cudaPrimAlgorithm(vertices, V, edges, E, outbound, inbound, weights);

    // now the real test run
    begin = steady_clock::now();
    for (int i=0; i<cntRuns; ++i) {
        // find MST solution
        cudaPrimAlgorithm(vertices, V, edges, E, outbound, inbound, weights);
    }
    end = steady_clock::now();
    runtime = (duration_cast<duration<double>>(end-begin)).count();

    mst.resize(V, V-1, g.is_directed());
    for (uint32_t i = 0; i < V-1; ++i) {
        mst.set(outbound[i], inbound[i], (uint32_t) weights[i]);
    }

    delete[] vertices;
    delete[] edges;
    delete[] outbound;
    delete[] inbound;
    delete[] weights;

    // return as miliseconds per round
    return 1000.*runtime/cntRuns;    
}

double cuda_multi_runtime(const Graph& g, int cntRuns, Graph& mst) {
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
	cuda_multi_setup(g, inbound_vertices, outbound_vertices);

	// initialize solution arrays with +inf
	std::fill(outbound, outbound + V, UINT32_MAX);
	std::fill(inbound, inbound + V, UINT32_MAX);
	std::fill(weights, weights + V, UINT32_MAX);

	// allow for warm-up
	cuda_multi_prim_algorithm(V, E, outbound_vertices, inbound_vertices, outbound, inbound, weights);

	// now the real test run
	begin = steady_clock::now();
	for (int i = 0; i < cntRuns; ++i) {
		// initialize solution arrays with +inf
		std::fill(outbound, outbound + V, UINT32_MAX);
		std::fill(inbound, inbound + V, UINT32_MAX);
		std::fill(weights, weights + V, UINT32_MAX);
		// find MST solution
		cuda_multi_prim_algorithm(V, E, outbound_vertices, inbound_vertices, outbound, inbound, weights);
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
        uint64_t itseed = seed+i;
        generator(g, num_vertices, 0, weight_range, density, false, itseed);
        // run through all implementations and get runtime
        double runtime;

        ListGraph cpu_l_mst;
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
/*
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
*/
        // run through cuda implementation
        ListGraph cuda_mst;
        runtime = cudaRuntime(g, cntRuns, cuda_mst);
        // output to file 
        os << "cuda," << i
                << "," << itseed
                << "," << num_vertices
                << "," << density
                << "," << weight_range
                << "," << runtime
                << "," << cuda_mst.sum_weights()
                << std::endl;
        // run through cuda multi implementation
		ListGraph cuda_multi_mst;
		runtime = cuda_multi_runtime(g, cntRuns, cuda_multi_mst);
		// output to file 
		os << "cuda-multi," << i
			<< "," << itseed
			<< "," << num_vertices
			<< "," << density
			<< "," << weight_range
			<< "," << runtime
			<< "," << cuda_multi_mst.sum_weights()
			<< std::endl;
    }
}

int main(int argc, char* argv[]) {
	std::cout << "implementation,seed,vertices,density,weight_range,runtime,min" << std::endl;
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
