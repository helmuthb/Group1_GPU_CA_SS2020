#ifndef __GENERATOR_HPP
#define __GENERATOR_HPP

#include "graph.hpp"
#include <cstdint>

/**
 * Generate a graph with specified number of vertices, the range for the weights,
 * the desired edge density and whether directed or undirected.
 * 
 * The number of vertices generated is exactly given through the density.
 */
void generator(Graph& result, uint32_t num_nodes, int32_t min_weight, int32_t max_weight, float density, bool directed);

#endif