#ifndef __GENERATOR_HPP
#define __GENERATOR_HPP

#include "graph.hpp"

/**
 * Generate a graph with specified number of vertices, the range for the weights,
 * the desired edge density and whether directed or undirected.
 * 
 * The number of vertices generated is exactly given through the density.
 */
Graph generator(int num_nodes, int min_weight, int max_weight, float density, bool directed);

#endif