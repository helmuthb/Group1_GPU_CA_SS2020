#ifndef __CUDA_PRIM_HPP
#define __CUDA_PRIM_HPP

#include "graph.hpp"
#include <bitset>
#include <vector>
#include <thrust/host_vector.h>

/**
 * Prepare for cuda version of Prim's algorith
 */
void cuda2Setup(const Graph& g, uint2 *vertices, uint2 *edges);

/**
 * Prim's algorithm
 */
void cuda2PrimAlgorithm(uint2 *vertices, uint32_t num_vertices,
                        uint2 *edges, uint32_t num_edges,
                        uint32_t *outbound, uint32_t *inbound, uint32_t *weights);

#endif
