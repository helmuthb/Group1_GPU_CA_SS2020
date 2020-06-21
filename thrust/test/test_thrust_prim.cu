#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "thrust_prim.hpp"
#include "list_graph.hpp"
#include "generator.hpp"
#include <set>

TEST_CASE("Thrust prim for tiny graph") {
    ListGraph g(6, false);
    g.set(0,1,1);
    g.set(0,2,3);
    g.set(0,5,2);
    g.set(1,2,5);
    g.set(1,3,1);
    g.set(2,3,2);
    g.set(2,4,1);
    g.set(3,4,4);
    g.set(4,5,5);
    uint32_t V = g.num_vertices();
    uint32_t E = g.num_edges();
    thrust::host_vector<uint2> vertex_adjacent_count_index(V);
    thrust::host_vector<uint2> edge_target_weight(2*E);
    thrust::host_vector<uint32_t> mst_out(V);
    thrust::host_vector<uint32_t> mst_in(V);
    thrust::host_vector<uint32_t> mst_weight(V);
    thrustSetup(g, vertex_adjacent_count_index, edge_target_weight);
    thrustPrimAlgorithm(vertex_adjacent_count_index, edge_target_weight, mst_out, mst_in, mst_weight, V, E);
    // sum up weights
    int w = 0;
    for (int i=0; i<V-1; ++i) {
        w += mst_weight[i];
    }
    CHECK(w == 7);
    // check correct weight from original graph
    for (int i=0; i<V-1; ++i) {
        int f = mst_out[i];
        int t = mst_in[i];
        std::cout << f << "->" << t << " (" << mst_weight[i] << ")" << std::endl;
        CHECK(mst_weight[i] == g(f, t));
    }
    // check reachability of each node
    std::set<int> mst_nodes;
    mst_nodes.insert(mst_out[0]);
    for (int i=0; i<V-1; ++i) {
        // check if source node already in mst nodes
        CHECK(mst_nodes.count(mst_out[i]) == 1);
        // add new target node
        mst_nodes.insert(mst_in[i]);
    }
    CHECK(mst_nodes.size() == V);
}

TEST_CASE("Thrust prim for micro graph") {
    ListGraph g(3, false);
    g.set(0,1,1);
    g.set(0,2,2);
    g.set(1,2,3);
    uint32_t V = g.num_vertices();
    uint32_t E = g.num_edges();
    thrust::host_vector<uint2> vertex_adjacent_count_index(V);
    thrust::host_vector<uint2> edge_target_weight(2*E);
    thrust::host_vector<uint32_t> mst_out(V);
    thrust::host_vector<uint32_t> mst_in(V);
    thrust::host_vector<uint32_t> mst_weight(V);
    thrustSetup(g, vertex_adjacent_count_index, edge_target_weight);
    thrustPrimAlgorithm(vertex_adjacent_count_index, edge_target_weight, mst_out, mst_in, mst_weight, V, E);
    // sum up weights
    int w = 0;
    for (int i=0; i<V-1; ++i) {
        w += mst_weight[i];
    }
    CHECK(w == 3);
    // check correct weight from original graph
    for (int i=0; i<V-1; ++i) {
        int f = mst_out[i];
        int t = mst_in[i];
        CHECK(mst_weight[i] == g(f, t));
    }
    // check reachability of each node
    std::set<int> mst_nodes;
    mst_nodes.insert(mst_out[0]);
    for (int i=0; i<V-1; ++i) {
        // check if source node already in mst nodes
        CHECK(mst_nodes.count(mst_out[i]) == 1);
        // add new target node
        mst_nodes.insert(mst_in[i]);
    }
    CHECK(mst_nodes.size() == V);
}

TEST_CASE("Thrust prim for large graph") {
    ListGraph g(1000, false);
    generator(g, 1000, 2, 10, 0.5, false);
    int max_weight = 10 * 999;
    uint32_t V = g.num_vertices();
    uint32_t E = g.num_edges();
    thrust::host_vector<uint2> vertex_adjacent_count_index(V);
    thrust::host_vector<uint2> edge_target_weight(2*E);
    thrust::host_vector<uint32_t> mst_out(V);
    thrust::host_vector<uint32_t> mst_in(V);
    thrust::host_vector<uint32_t> mst_weight(V);
    thrustSetup(g, vertex_adjacent_count_index, edge_target_weight);
    thrustPrimAlgorithm(vertex_adjacent_count_index, edge_target_weight, mst_out, mst_in, mst_weight, V, E);
    // sum up weights
    int w = 0;
    for (int i=0; i<V-1; ++i) {
        w += mst_weight[i];
    }
    CHECK(w <= max_weight);
    // check correct weight from original graph
    int wrong_weights = 0;
    for (int i=0; i<V-1; ++i) {
        int f = mst_out[i];
        int t = mst_in[i];
        if (mst_weight[i] != g(f, t)) ++wrong_weights;
    }
    CHECK(wrong_weights == 0);
    // check reachability of each node
    std::set<int> mst_nodes;
    mst_nodes.insert(mst_out[0]);
    int nodes_missing = 0;
    for (int i=0; i<V-1; ++i) {
        // check if source node already in mst nodes
        if (mst_nodes.count(mst_out[i]) != 1) ++nodes_missing;
        // add new target node
        mst_nodes.insert(mst_in[i]);
    }
    CHECK(nodes_missing == 0);
    CHECK(mst_nodes.size() == V);
}
