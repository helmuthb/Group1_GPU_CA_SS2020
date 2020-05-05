#include <stdexcept>
#include <iostream>
#include <bits/stdc++.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "graph.hpp"
#include "generator.hpp"

const static int MAX_NODES = 100000;

/**
 * Helper function: return random number from 0 to 1 (exclusive)
 */
static double rnum() {
    return double(rand()) / RAND_MAX;
}

/**
 * Generate a graph with defined number of vertices, range of edge weight, density.
 */
Graph generator(int num_vertices, int min_weight, int max_weight, float density, bool directed) {
    Graph g(num_vertices, directed);
    // check num_nodes
    if (num_vertices <= 0 || num_vertices > MAX_NODES) {
        throw new std::out_of_range("Maximum number of nodes exceeded");
    }
    // check weight range
    if (max_weight <= min_weight || min_weight < 0 || max_weight >= Graph::WEIGHT_INFTY) {
        throw new std::out_of_range("Weight range exceeded");
    }
    // check density
    if (density <= 0 || density >=1) {
        throw new std::out_of_range("Density range exceeded");
    }
    int weight_range = max_weight - min_weight + 1;
    // loop through possible edges, i < j
    for (int i=0; i<num_vertices; i++) {
        int j0 = (g.is_directed() ? 0 : i+1);
        for (int j=j0; j<num_vertices; j++) {
            bool edge_exists = rnum() < density;
            unsigned int w = edge_exists ? min_weight + rnum()*weight_range : Graph::WEIGHT_INFTY;
            g.set(i, j, w);
        }
    }
    return g;
}
