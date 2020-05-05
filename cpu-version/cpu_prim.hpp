#ifndef __CPU_PRIM_HPP
#define __CPU_PRIM_HPP

#include "graph.hpp"
#include <bitset>
#include <vector>

/**
 * Find nearest vertex which is not yet in the set of
 * vertices identified by the bitset v
 */
int nearestVertex(const Graph& g, const int d[], const bool v[]);

/**
 * Prim's algorithm
 */
Graph primAlgorithm(const Graph& g);

#endif