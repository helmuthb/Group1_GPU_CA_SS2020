#include <stdexcept>
#include <iostream>
#include <random>
// #include <bits/stdc++.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "graph.hpp"
#include "generator.hpp"

const static int MAX_NODES = 100000;

/**
 * Generate a graph with defined number of vertices, range of edge weight, density.
 */
Graph generator(int num_vertices, int min_weight, int max_weight, float density, bool directed) {
    Graph g(num_vertices, directed);
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
    if (density <= 0 || density >=1) {
        throw new std::out_of_range("Density range exceeded");
    }
    // calculate desired number of edges
    long long nv_l = (long long)num_vertices;
    long long max_edges = directed?(nv_l*(nv_l-1)):(nv_l*(nv_l-1)/2);
    int num_edges = max_edges*density;
    // Using algorithm of Floyd to get this number of edges
    for (long long j=max_edges-num_edges; j<max_edges; j++) {
        // draw random v in range (0, j)
        long long v = std::uniform_int_distribution<long long>(0, j)(gen);
        // weight we will use
        unsigned int w = std::uniform_int_distribution<>(min_weight, max_weight)(gen);
        // is the edge v not yet added?
        if (g(v) == Graph::WEIGHT_INFTY) {
            g.set(v, w);
        }
        // otherwise add edge j - it cannot have been added yet
        else {
            g.set(j, w);
        }
    }
    return g;
}
