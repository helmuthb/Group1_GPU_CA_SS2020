#ifndef __CUDA_PRIM_HPP
#define __CUDA_PRIM_HPP

#include "graph.hpp"
#include <bitset>
#include <vector>
#include <thrust/host_vector.h>

/**
 * Prepare for cuda version of Prim's algorith
 */
void cudaSetup(const Graph& g, uint2 *&inbound_vertices, uint2 *&outbound_vertices, uint2 *&shape);

/**
 * Prim's algorithm
 */
void cudaPrimAlgorithm(uint2 *inbound_vertices, uint2 *outbound_vertices, uint2 *shape,
                       uint32_t *inbound, uint32_t *outbound, uint32_t *weights);

#endif