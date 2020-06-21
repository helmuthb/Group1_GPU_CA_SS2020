#ifndef __CUDA_MULTI_PRIM_HPP
#define __CUDA_MULTI_PRIM_HPP

#define SHM_FACTOR 2

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include "list_graph.hpp"
#include "generator.hpp"
#include "graph.hpp"

#include <chrono>

using namespace std;

/**
 * Prepare for cuda-multi version of Prim's algorith
 */
void cuda_multi_setup(const Graph& g, uint2 *&inbound_vertices, uint2 *&outbound_vertices);

/**
 * Prim's algorithm
 */
void cuda_multi_prim_algorithm(uint32_t num_vertices, uint32_t num_edges, uint2 *outbound_vertices, uint2 *inbound_vertices, uint32_t *outbound, uint32_t *inbound, uint32_t *weights);

#endif
