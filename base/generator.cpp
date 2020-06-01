#include <stdexcept>
#include <iostream>
#include <random>
#include <cstdint>
#include "graph.hpp"
#include "generator.hpp"

const static int MAX_NODES = 100000;

/**
 * Generate a graph with defined number of vertices, range of edge weight, density.
 */
void generator(Graph& g, uint32_t num_vertices, int32_t min_weight, int32_t max_weight, float density, bool directed) {
    // random number generator
    std::random_device rd;
    std::mt19937_64 gen(rd());

    // check num_nodes
    if (num_vertices <= 0 || num_vertices > MAX_NODES) {
        throw new std::out_of_range("Maximum number of nodes exceeded");
    }
    // check weight range
    if (max_weight <= min_weight || min_weight < 0 || max_weight >= Graph::WEIGHT_INFTY) {
        throw new std::out_of_range("Weight range exceeded");
    }
    // check density
    // Note that an MST requires a connected graph, that is: every node of the
    // graph must be reachable. For an undirected graph, the minimum number of
    // edges is |V-1|, which corresponds to a minimum density of 2/v.
    if (density < 2.0/num_vertices) {
        throw new std::out_of_range("Density insufficient to connect graph");
    } else if (density > 1) {
        throw new std::out_of_range("Density range exceeded");
    }
    // calculate desired number of edges
    uint64_t nv_l = (uint64_t)num_vertices;
    uint64_t max_edges = directed?(nv_l*(nv_l-1)):(nv_l*(nv_l-1)/2);
    uint64_t num_edges = max_edges*density;
    // resize graph
    g.resize(num_vertices, num_edges, directed);
    // Using algorithm of Floyd to get this number of edges
    for (uint64_t j=max_edges-num_edges; j<max_edges; j++) {
        // draw random v in range (0, j)
        uint64_t v = std::uniform_int_distribution<uint64_t>(0, j)(gen);
        // weight we will use
        int32_t w = std::uniform_int_distribution<>(min_weight, max_weight)(gen);
        // is the edge v not yet added?
        if (g(v) == Graph::WEIGHT_INFTY) {
            g.set(v, w);
        }
        // otherwise add edge j - it cannot have been added yet
        else {
            g.set(j, w);
        }
    }
}
