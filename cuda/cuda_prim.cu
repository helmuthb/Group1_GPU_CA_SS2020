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

// Threads per block to use
#define BLOCKSIZE 1024


//
// Initialize the compact adjacency list representation (Wang et al.)
//
// Refer to the report for a detailed explanation of this data structure.
//
// The input graph is generated using our own graph generator, which can be
// found in base/.
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
// Kernel implementing the weight update primitive
//
// Uses the compact adjacency list as read-only input, and writes to the MST
// data structure.
//
// Each thread accesses only one "row" of the MST data structure, so there is
// no need to synchronize anything.
//
// current_vertex points to the ID of the vertex from which the new paths are to be
// checked, and num_fixed is position of the "remainder" window.
//  
__global__ void mst_update(uint2 *vertices, uint2 *edges,
                           uint32_t *outbound, uint32_t *weights, uint32_t *v2i_map,
                           uint32_t *current_vertex, uint32_t num_fixed)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t offset = vertices[*current_vertex].y;
    uint32_t count = vertices[*current_vertex].x;

    // We only need as many threads as new node has edges
    // => the denser a graph, the better we can exploit parallization!
    if (idx < count) {
        uint2 edge = edges[offset+idx];
        uint32_t vertex = edge.x;
        uint32_t weight = edge.y;
        // Get the current postion of this vertex in the MST data structure
        // The map tracks where each vertex is currently located. The location
        // may change after a swap operation.
        uint32_t vertex_idx = v2i_map[vertex];

        // Only check this vertex if it's not already part of the MST (it's
        // location is larger or equal to the already fixed part)
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
//   Use NULL in the first step of the reduction    => SETS         the index in tmp_minindices
//   Use non-NULL as input to the second reduction  => CARRIES OVER the index in tmp_minindices
//
__global__ void mst_minreduce(uint32_t *indices, uint32_t *weights,
                              uint32_t *tmp_minindices, uint32_t *tmp_minweights,
                              uint32_t num_remaining)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    // TODO: This is the vanilla, totally un-optimized version of the
    // reduction! Once this is working, adapt as per the NVIDIA slides

    // Store the per-thread best index and minimum weight
    __shared__ uint32_t shm_minindices[BLOCKSIZE];
    __shared__ uint32_t shm_minweights[BLOCKSIZE];

    if (idx < num_remaining) {
        // Each thread loads one element from global to shared memory (indices optional)
        if (indices == NULL) {
            shm_minindices[threadIdx.x] = idx;
        } else {
            shm_minindices[threadIdx.x] = indices[idx];
        }
        shm_minweights[threadIdx.x] = weights[idx]; 

        __syncthreads();

        // Perform the reduction, as per NVIDIA guidelines
        for (uint32_t s = 1; s < blockDim.x; s *= 2) {
            uint32_t left = 2 * s * threadIdx.x;

            if (left < blockDim.x) { uint32_t right = left + s;
                // Only compare if the counterpart is still within bounds
                if (right + (blockDim.x * blockIdx.x) < num_remaining) {
                    // If the best weight is not already at position ti, move it there
                    if (shm_minweights[right] < shm_minweights[left]) {
                        shm_minindices[left] = shm_minindices[right];
                        shm_minweights[left] = shm_minweights[right];
                    }
                }
            }
            __syncthreads();
        }

        // The last active thread of the block writes the result back
        if (threadIdx.x == 0) {
            tmp_minindices[blockIdx.x] = shm_minindices[0];
            tmp_minweights[blockIdx.x] = shm_minweights[0];
        }
    }
}


//
// Kernel implementing the swap and next node selection primitive
//
// The best remaining edge will be moved to the start of the remainder window,
// and it's target vertex will become the new current vertex.
//
// (In the next iteration, the remainder window will be decreased by 1, thereby
// fixing this edge permanently).
//
__global__ void mst_swap_and_next(uint32_t *outbound, uint32_t *inbound, uint32_t *weights,
                                  uint32_t *v2i_map, uint32_t *tmp_minindices,
                                  uint32_t *current_vertex)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t best = tmp_minindices[0];

    // We only need one thread for this
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

            uint32_t mapA  = v2i_map[inA];
            uint32_t mapB  = v2i_map[inB];

            v2i_map[inA]   = mapB;
            v2i_map[inB]   = mapA;
        }

        // Update the current vertex for the next iteration
        *current_vertex = inbound[0];
    }
}


//
// Entry point for CUDA Prim's Algorithm
//
// This uses:
//   * Compact Adjacency List as proposed by Wang et al., based on Harish et al.
//   * MST data structure as proposed by Wang et al.
//   * We extend this 3-part MST data structure with a 4th helper, the
//     vertex-to-index map, which keeps track of a location of a vertex after
//     it has been moved.
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

    // Data structures in device memory
    uint2 *d_vertices, *d_edges;
    uint32_t *d_outbound, *d_inbound, *d_weights;
    // Temporary helpers and results storage
    uint32_t *d_v2i_map, *d_tmp_minindices, *d_tmp_minweights, *d_current_vertex;

    // Sanity checks
    if (BLOCKSIZE == 1) {
        throw new std::out_of_range("BLOCKSIZE must be greater than 1");
    } else if (ceil(log2(BLOCKSIZE)) != floor(log2(BLOCKSIZE))) {
       throw new std::out_of_range("BLOCKSIZE must be a power of 2");
    }
    // Total number of blocks needed to process all edges (one thread per edge)
    uint32_t blocks_total = static_cast<uint32_t>(std::ceil(static_cast<float>(num_vertices-1) / BLOCKSIZE));
    if (blocks_total > BLOCKSIZE) {
        throw new std::out_of_range("Cannot reduce more than BLOCKSIZE blocks");
    }

    // Allocate memory for the data structures in device memory
    cudaMalloc(&d_vertices,       num_vertices     * sizeof(uint2));
    cudaMalloc(&d_edges,          num_edges        * sizeof(uint2));
    cudaMalloc(&d_outbound,       (num_vertices-1) * sizeof(uint32_t));
    cudaMalloc(&d_inbound,        (num_vertices-1) * sizeof(uint32_t));
    cudaMalloc(&d_weights,        (num_vertices-1) * sizeof(uint32_t));
    cudaMalloc(&d_v2i_map,        num_vertices     * sizeof(uint32_t));
    cudaMalloc(&d_tmp_minindices, blocks_total     * sizeof(uint32_t));
    cudaMalloc(&d_tmp_minweights, blocks_total     * sizeof(uint32_t));
    cudaMalloc(&d_current_vertex, 1                * sizeof(uint32_t));

    // Transfer inputs to device memory
    cudaMemcpy(d_vertices,  vertices,  num_vertices     * sizeof(uint2),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges,     edges,     num_edges        * sizeof(uint2),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_outbound,  outbound,  (num_vertices-1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inbound,   inbound,   (num_vertices-1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights,   weights,   (num_vertices-1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_tmp_minindices, 0,    blocks_total     * sizeof(uint32_t));
    cudaMemset(d_tmp_minweights, 0,    blocks_total     * sizeof(uint32_t));
    cudaMemset(d_current_vertex, 0,    1                * sizeof(uint32_t));

    // We set the first two positions to 0, and fill the rest of the positions
    // with the sequence in inbound (1 .. |V|-1).
    cudaMemset(d_v2i_map,        0,    2                * sizeof(uint32_t));
    cudaMemcpy(d_v2i_map+2, inbound,   (num_vertices-2) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    //
    // Outer loop:
    // The iteration variable is the offset of the "remainder" window:
    //   * Any edge before this offset is fixed into the MST
    //   * Any edge after this offset is just a candidate
    // The window decreases by 1 every iteration, until all edges have been fixed.
    //
    for (uint32_t num_fixed = 0; num_fixed < num_vertices - 1; ++num_fixed) {
        uint32_t num_remaining    = num_vertices - 1 - num_fixed;
        uint32_t blocks_remaining = static_cast<uint32_t>(std::ceil(static_cast<float>(num_remaining) / BLOCKSIZE));

        mst_update <<<blocks_total, BLOCKSIZE>>> (d_vertices, d_edges,
                                                  d_outbound, d_weights, d_v2i_map,
                                                  d_current_vertex, num_fixed);

        // Min-Reduce Level 1: minimum per block, stored in temporary result
        // The NULL in this level means that the kernel will assign a running
        // index for us.
        mst_minreduce <<<blocks_remaining, BLOCKSIZE>>> (NULL, d_weights+num_fixed,
                                                         d_tmp_minindices, d_tmp_minweights,
                                                         num_remaining);

        // Min-Reduce Level 2:
        // If the previous level needed more than one block, find the minimum
        // of all blocks. 
        if (blocks_remaining > 1) {
            mst_minreduce <<<1, blocks_remaining>>> (d_tmp_minindices, d_tmp_minweights,
                                                     d_tmp_minindices, d_tmp_minweights,
                                                     blocks_remaining);
        }

        // Swap the best edge to the start (if necessary), and update the current vertex
        mst_swap_and_next <<<1, 1>>> (d_outbound+num_fixed, d_inbound+num_fixed, d_weights+num_fixed,
                                      d_v2i_map, d_tmp_minindices,
                                      d_current_vertex);
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
    cudaFree(d_tmp_minindices);
    cudaFree(d_tmp_minweights);
    cudaFree(d_current_vertex);
}
