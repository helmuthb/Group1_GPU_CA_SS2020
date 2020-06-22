#ifndef __THRUST_PRIM_HPP
#define __THRUST_PRIM_HPP

#include "graph.hpp"
#include <bitset>
#include <vector>
#include <thrust/host_vector.h>

/**
 * Prepare for thrust version of Prim's algorith
 */
void thrustSetup(const Graph& g, thrust::host_vector<uint2> &vertex_adjacent_count_index, thrust::host_vector<uint2> &edge_target_weight);

/**
 * Prim's algorithm
 */
void thrustPrimAlgorithm(const thrust::host_vector<uint2> &vertex_adjacent_count_index, const thrust::host_vector<uint2> &edge_target_weight,
                         thrust::host_vector<uint32_t> &mst_out, thrust::host_vector<uint32_t> &mst_in,
                         thrust::host_vector<uint32_t> &mst_weight,
                         uint32_t V, uint32_t E);

#endif