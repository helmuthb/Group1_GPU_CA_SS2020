#ifndef __THRUST_PRIM_HPP
#define __THRUST_PRIM_HPP

#include "graph.hpp"
#include <bitset>
#include <vector>
#include <thrust/host_vector.h>

/**
 * Prepare for thrust version of Prim's algorith
 */
void thrustPrepare(const Graph& g, thrust::host_vector<uint32_t>* num_neighbors, thrust::host_vector<uint32_t>* idx_edges,
                   thrust::host_vector<uint32_t>* target, thrust::host_vector<int32_t>* weight);

/**
 * Prim's algorithm
 */
void thrustPrimAlgorithm(thrust::host_vector<uint32_t>* num_edges, thrust::host_vector<uint32_t>* idx_edges,
                          thrust::host_vector<uint32_t>* target, thrust::host_vector<int32_t>* weight,
                          thrust::host_vector<uint32_t>* predecessors);

#endif