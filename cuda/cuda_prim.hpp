#ifndef __CUDA_PRIM_HPP
#define __CUDA_PRIM_HPP

#include "graph.hpp"
#include <bitset>
#include <vector>
#include <thrust/host_vector.h>

/**
 * Prepare for cuda version of Prim's algorith
 */
void cudaSetup(const Graph& g, uint2 *&vertex_adjacent_count_index, uint2 *&edge_target_weight);

/**
 * Prim's algorithm
 */
void cudaPrimAlgorithm(uint2 *vertex_adjacent_count_index, uint2 *edge_target_weight,
                       uint32_t *mst_out, uint32_t *mst_in, uint32_t *mst_weight,
                       uint32_t V, uint32_t E);

#endif