#ifndef __CPU_PRIM_HPP
#define __CPU_PRIM_HPP

#include "graph.hpp"
#include <bitset>
#include <vector>

/**
 * Internal function: find the nearest vertex outside the
 * subgraph identified so far
 */
int cpuNearestVertex(const Graph& g, const int d[], const bool v[]);

/**
 * Prim's algorithm
 */
Graph cpuPrimAlgorithm(const Graph& g);

#endif