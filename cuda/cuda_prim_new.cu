//
// CUDA implemenation of Prim's Minimum Spanning Tree Algorithm
//
//
// Please refer to the report for documentation on all the data structures used
// here, as well as an outline of the implementation.
//


#include <cmath>

#include "cuda_prim.hpp"


//////////////////////////
// Options
//////////////////////////

#define BLOCKSIZE 256


//
// Initialize the compact adjacency list representation (Wang et al.)
// 
void cudaSetup(const Graph& g, uint2 *vertices, uint2 *edges)
{
    uint32_t num_vertices = g.num_vertices();

    // Calculate data for each vertex, and the edges to its neighbors 
    for (uint32_t v = 0; v < num_vertices; ++v) {
        std::vector<EdgeTarget> neighbors;
        g.neighbors(v, neighbors);

        // Store vertex neighbor count and offset
        vertices[v].x = neighbors.size();
        vertices[v].y = 0;
        if (v == 0) {
            // Base case
            vertices[v].y = 0;
        } else {
            // Current offset = previous offset + number of previous nodes
            vertices[v].y = vertices[v-1].y + vertices[v-1].x;
        }

        // Store each edge, starting at the previously computed offset
        uint32_t idx = vertices[v].y;
        for (auto nb = neighbors.begin(); nb < neighbors.end(); ++nb) {
            // Store the neighbor vertex ID, and the weight of the edge
            edges[idx].x = nb->vertex_to;
            edges[idx].y = nb->weight;
            idx++;
        }
    }
}


//
// Kernel implementing the swap operation
//
//
__global__ void mst_swap_and_next(uint32_t *outbound, uint32_t *inbound, uint32_t *weights,
                                  uint32_t *v2i_map, uint32_t *tmp_best, uint32_t *current_vertex)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t best = *tmp_best;
    if (idx == 0) {
        // No need to swap if the best edge is already at the front
        if (best != 0) {
            uint32_t outA = outbound[0];
            uint32_t inA  = inbound[0];
            uint32_t wA   = weights[0];
            uint32_t outB = outbound[best];
            uint32_t inB  = inbound[best];
            uint32_t wB   = weights[best];

            outbound[0]    = outB;
            inbound[0]     = inB;
            weights[0]     = wB;
            outbound[best] = outA;
            inbound[best]  = inA;
            weights[best]  = wA;

            uint32_t mapA = v2i_map[inA];
            uint32_t mapB = v2i_map[inB];
            v2i_map[inA]   = mapB;
            v2i_map[inB]   = mapA;
        }

        *current_vertex = inbound[0];
    }
}


//
// Kernel implementing the weight update primitive
//
// Uses the compact adjacency list as read-only input, and writes to the four
// MST data structures. Each thread accesses only one "row" of the MST data
// structure, so there is no need to synchronize anything.
//
// current_vertex points to the ID of the vertex from which the new paths are to be
// checked, and num_fixed is position of the "remainder" window.
//  
__global__ void mst_update(uint2 *vertices, uint2 *edges,
                           uint32_t *outbound, uint32_t *inbound, uint32_t *weights,
                           uint32_t *v2i_map, uint32_t *current_vertex, uint32_t num_fixed)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t offset = vertices[*current_vertex].y;
    uint32_t count  = vertices[*current_vertex].x;

    // We only need as many threads as new node has edges
    if (idx < count) {
        uint2 edge = edges[offset+idx];
        uint32_t vertex = edge.x;
        uint32_t weight = edge.y;
        uint32_t vertex_idx = v2i_map[vertex];

        // Only check this vertex if it is the "remainder" window
        if (num_fixed <= vertex_idx) {
            // If the new weight is better, replace the existing edge
            if (weight < weights[vertex_idx]) {
                outbound[vertex_idx] = *current_vertex;
                weights[vertex_idx] = weight;
            }
        }
    }
}


//
// Kernel implementing the min reduction primitive
//
// indices:
//   Use NULL in the first step of the reduction    => SETS the index
//   Use non-NULL as input to the second reduction  => CARRIES over the index
//
__global__ void mst_minweight(uint32_t *indices, uint32_t *weights,
                              uint32_t *tmp_best, uint32_t *tmp_minweights,
                              uint32_t num_remaining)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    // TODO: This is the vanilla, totally un-optimized version of the
    // reduction! Once this is working, adapt as per the NVIDIA slides

    // Store the per-thread best index and minimum weight
    __shared__ uint32_t shm_best[BLOCKSIZE];
    __shared__ uint32_t shm_minweights[BLOCKSIZE];

    if (idx < num_remaining) {
        // Each thread loads one element from global to shared memory (indices optional)
        if (indices == NULL) {
            shm_best[threadIdx.x] = idx;
        } else {
            shm_best[threadIdx.x] = indices[idx];
        }
        shm_minweights[threadIdx.x] = weights[idx]; 

        __syncthreads();

        // Perform the reduction, as per NVIDIA guidelines
        for (uint32_t s = 1; s < blockDim.x; s *= 2) {
            uint32_t left = 2 * s * threadIdx.x;

            if (left < blockDim.x) {
                uint32_t right = left + s;
                // Conditions:
        //  1. Don't test outside of the block (eg uneven block size)
        //  2. Input size might not be power of two, so cut off  appropriately
                if (right < blockDim.x && right + (blockDim.x * blockIdx.x) < num_remaining) {
                    // If the best weight is not already at position ti, move it there
                    if (shm_minweights[right] < shm_minweights[left]) {
                        shm_best[left] = shm_best[right];
                        shm_minweights[left] = shm_minweights[right];
                    }
                }
            }
            __syncthreads();
        }

        // The last active thread of the block writes the result back
        if (threadIdx.x == 0) {
            tmp_best[blockIdx.x] = shm_best[0];
            tmp_minweights[blockIdx.x] = shm_minweights[0];
        }
    }
}


//
// Entry point for CUDA Prim's Algorithm
//
// This uses:
//   * Compact Adjacency List as proposed by Wang et al., based on Harish et al.
//   * MST data structure as proposed by Wang et al.
//
void cudaPrimAlgorithm(uint2 *vertices, uint32_t num_vertices,
                       uint2 *edges, uint32_t num_edges,
                       uint32_t *outbound, uint32_t *inbound, uint32_t *weights) {

    // Initialize the MST data structure
    for (uint32_t i = 0; i < num_vertices - 1; ++i) {
        outbound[i] = 0;
        inbound[i] = i + 1;
        weights[i] = Graph::WEIGHT_INFTY;
    }

    uint32_t max_edges = 0;
    for (uint32_t i = 0; i < num_vertices; ++i) {
	    if (vertices[i].x > max_edges) {
		    max_edges = vertices[i].x;
	    }
    }
    printf("max_edges=%u\n", max_edges);

    // Data structures in device memory
    uint2 *d_vertices, *d_edges;
    uint32_t *d_outbound, *d_inbound, *d_weights;
    // Temporary helpers and results storage
    uint32_t *d_v2i_map, *d_tmp_best, *d_tmp_minweights, *d_current_vertex;

    // Total number of blocks needed to process all edges (one thread per edge)
    uint32_t total_blocks = static_cast<uint32_t>(std::ceil(static_cast<float>(num_vertices-1) / BLOCKSIZE));
    if (total_blocks > 1024) {
        throw new std::out_of_range("Cannot reduce more than 1024 blocks");
    }

    // Allocate memory for the data structures in device memory
    cudaMalloc(&d_vertices,       num_vertices     * sizeof(uint2));
    cudaMalloc(&d_edges,          num_edges        * sizeof(uint2));
    cudaMalloc(&d_outbound,       (num_vertices-1) * sizeof(uint32_t));
    cudaMalloc(&d_inbound,        (num_vertices-1) * sizeof(uint32_t));
    cudaMalloc(&d_weights,        (num_vertices-1) * sizeof(uint32_t));
    cudaMalloc(&d_v2i_map,        num_vertices     * sizeof(uint32_t));
    cudaMalloc(&d_tmp_best,       total_blocks     * sizeof(uint32_t));
    cudaMalloc(&d_tmp_minweights, total_blocks     * sizeof(uint32_t));
    cudaMalloc(&d_current_vertex, 1                * sizeof(uint32_t));

    // Transfer inputs to device memory
    cudaMemcpy(d_vertices,  vertices,  num_vertices     * sizeof(uint2),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges,     edges,     num_edges        * sizeof(uint2),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_outbound,  outbound,  (num_vertices-1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inbound,   inbound,   (num_vertices-1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights,   weights,   (num_vertices-1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2i_map+2, inbound,   (num_vertices-2) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_v2i_map,        0,    2                * sizeof(uint32_t));
    cudaMemset(d_tmp_best,       0,    total_blocks     * sizeof(uint32_t));
    cudaMemset(d_tmp_minweights, 0,    total_blocks     * sizeof(uint32_t));
    cudaMemset(d_current_vertex, 0,    1                * sizeof(uint32_t));

    for (uint32_t remaining_offset = 0; remaining_offset < num_vertices - 1; ++remaining_offset) {
        uint32_t num_remaining        = num_vertices - 1 - remaining_offset;
        uint32_t num_remaining_blocks = static_cast<uint32_t>(std::ceil(static_cast<float>(num_remaining) / BLOCKSIZE));

        mst_update <<<total_blocks, BLOCKSIZE>>> (
                d_vertices, d_edges,
                d_outbound, d_inbound, d_weights,
                d_v2i_map, d_current_vertex, remaining_offset);

        // Invoke 1: minimum per block, stored in temporary result
        mst_minweight <<<num_remaining_blocks, BLOCKSIZE>>> (
                // Let minweight index the data
                NULL, 
                // Each iteration, we move forward in the MST list
                d_weights+remaining_offset,
                // But not in the temporary results list!
                d_tmp_best, d_tmp_minweights,
                num_remaining);

        // Invoke 2:
        // If we have more than one block, find minimum of all blocks
        if (total_blocks > 1) {
            mst_minweight <<<1, total_blocks>>> (
                    d_tmp_best, d_tmp_minweights,
                    d_tmp_best, d_tmp_minweights,
                    total_blocks);
        }

        // If the best edge is not at the beginning, we must swap edges
        mst_swap_and_next <<<1, 1>>> (
                d_outbound+remaining_offset, d_inbound+remaining_offset, d_weights+remaining_offset,
                d_v2i_map, &d_tmp_best[0], d_current_vertex);
    } // Outer loop

    // Copy the results back to host memory
    cudaMemcpy(outbound, d_outbound, (num_vertices-1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(inbound,  d_inbound,  (num_vertices-1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(weights , d_weights,  (num_vertices-1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_vertices);
    cudaFree(d_edges);
    cudaFree(d_inbound);
    cudaFree(d_outbound);
    cudaFree(d_weights);
    cudaFree(d_v2i_map);
    cudaFree(d_tmp_best);
    cudaFree(d_tmp_minweights);
    cudaFree(d_current_vertex);
}