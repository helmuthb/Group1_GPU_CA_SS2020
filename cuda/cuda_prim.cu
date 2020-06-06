#include "cuda_prim.hpp"
//
// Single-kernel (one block) implementation of CUDA Prim
//
// Because there is only one block, there is no need for a synchronization
// point among many blocks prior to the last reduction step. The final
// reduction of the one block, is the final reduction of the entire solution.
//

// Options
// Use this many threads per block
#define BLOCKSIZE 1024


//
// Initialize the compact adjacency list graph representation (Wang et al.)
//
//  shape.x = |V|
//  shape.y = |E|
//
// outbound vertices: for each vertex with number k
//      outbound_vertices[k].x = count of immediate neighbors
//      outbound_vertices[k].y = offset of edge list in inbound_vertices
//
// inbound_vertices: incoming edges ordered by vertex
//      inbound_vertices[*].x = id of other vertex
//      inbound_vertices[*].y = weight of edge
//
// So for each vertex k:
//     offset         = outbound_vertices[k].y
//     num_edges      = outbound_vertices[k].x
//     list_of_edges: = inbound_vertices[offset+0]
//                      inbound_vertices[offset+1]
//                      inbound_vertices[offset+..]
//                      inbound_vertices[offset+num_edges]
//
void cudaSetup(const Graph& g, uint2 *&inbound_vertices, uint2 *&outbound_vertices, uint2 *&shape) {
    shape = new uint2;
    shape->x = g.num_vertices();
    shape->y = g.num_edges();
    inbound_vertices = new uint2[shape->y * 2];
    outbound_vertices = new uint2[shape->x];
    uint32_t pos = 0;
    // for each vertex
    for (uint32_t v = 0; v < shape->x; ++v) {
        // get its neighbors
        std::vector<EdgeTarget> neighbors;
        g.neighbors(v, neighbors);
        // store neighbor count and offset
        outbound_vertices[v].x = neighbors.size();
        outbound_vertices[v].y = v == 0 ? 0 : outbound_vertices[v - 1].y + outbound_vertices[v-1].x;
        // for each incoming edge
        for (auto nb = neighbors.begin(); nb < neighbors.end(); ++nb) {
            // store the neighbor vertex ID, and the weight of the edge
            // (The graph is undirected, to vertex_to is always the other side)
            inbound_vertices[pos].x = nb->vertex_to;
            inbound_vertices[pos++].y = nb->weight;
        }
    }
}

//
// Single-kernel Prim implementation
//
// Parallelizes the Min-Reduction (inner loop) of Prim.
//
// Params
// ------
//    [INPUT] inbound_vertices, outbound_vertices, shape, current_node:
//      Graph data as preprocessed by cudaSetup() in device memory.
//      As a reminder, shape.x = |V| and shape.y = |E|
//
//    [OUTPUT] inbound, outbound, weights
//      List of all edges of the MST as determined by this kernel.
//      For every edge i, if weights[i] is set, then the edge between
//      inbound[i] and outbound[i] is part of the MST.
//  
__global__ void mst(uint2 *inbound_vertices, uint2 *outbound_vertices, uint2 *shape,
                    uint32_t *inbound, uint32_t *outbound, uint32_t *weights, uint32_t *current_node)
{
    // The result of each individual thread's Min-Reduction is stored in shared memory
    __shared__ uint2 shm[BLOCKSIZE];

    // Which we initialize to the maximum permitted value
    if (threadIdx.x < BLOCKSIZE) {
        shm[threadIdx.x].y = UINT32_MAX;
    }

    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int num_nodes = shape->x;

    // Determine the minimum for each node
    for (int i = 0; i < num_nodes - 1; i++) {
        // As per cudaSetup() above, in outbound_vertices, .y = offset and .x = number of edges
        // So the edges of node i are
        //    inbound_vertices[start_index+0]
        //    inbound_vertices[start_index+1]
        //    inbound_vertices[start_index+...]
        //    inbound_vertices[start_index+end_index]
        // and each edge .x = the other node, and .y = the weight
        uint32_t start_index = outbound_vertices[*current_node].y;
        uint32_t end_index = start_index + outbound_vertices[*current_node].x;

        if (idx >= start_index && idx < end_index) {
            if (inbound_vertices[idx].y < weights[inbound_vertices[idx].x]) {
                weights[inbound_vertices[idx].x] = inbound_vertices[idx].y;
                outbound[inbound_vertices[idx].x] = *current_node;
            }
        }
        __syncthreads();

        shm[threadIdx.x].y = idx < max_nodes && inbound[idx] > max_nodes ? weights[idx] : UINT32_MAX;
        shm[threadIdx.x].x = idx;

        __syncthreads();

        for (int j = BLOCKSIZE; j > 1; j /= 2) {
            if (threadIdx.x < j / 2) {
                if (shm[threadIdx.x].y > shm[threadIdx.x + j / 2].y) {
                    shm[threadIdx.x].x = shm[threadIdx.x + j / 2].x;
                    shm[threadIdx.x].y = shm[threadIdx.x + j / 2].y;
                }
            }
        }
        
        if (idx == 0) {
            outbound[*current_node] = outbound[shm[0].x];
            weights[*current_node] = shm[0].y;
            inbound[*current_node] = shm[0].x;
            *current_node = shm[0].x;
            weights[*current_node] = UINT32_MAX;
        }
        __syncthreads();
    }
}

//
// Entry point for CUDA Prim
//
// Params
// ------
//    [INPUT] inbound_vertices, outbound_vertices, shape:
//      Graph data as preprocessed by cudaSetup()
//
//    [OUTPUT] inbound, outbound, weights:
//      List of all edges. For every edge i, if weights[i] is set, then the
//      edge between inbound[i] and outbound[i] is part of the MST.
//  
void cudaPrimAlgorithm(uint2 *inbound_vertices, uint2 *outbound_vertices, uint2 *shape,
                       uint32_t *inbound, uint32_t *outbound, uint32_t *weights) {

    const uint32_t V = shape->x;
    const uint32_t E = shape->y;
    // Input data structures
    uint2 *d_inbound_vertices, *d_outbound_vertices, *d_shape;
    // Output data structures
    uint32_t *d_outbound, *d_inbound, *d_weights;
    uint32_t current_node = 0, *d_current_node;

    // Our weights will signify whether an edge is part of the MST or not. We
    // begin with all edges not set
    // TODO: Check whether this can be replaced with cudaMemset, which might be
    // faster
    std::fill_n(weights, V, Graph::WEIGHT_INFTY);

    // Allocate memory for the input data structures
    cudaMalloc(&d_inbound_vertices, E * 2 * sizeof(uint2));
    cudaMalloc(&d_outbound_vertices, V * sizeof(uint2));
    cudaMalloc(&d_shape, sizeof(uint2));

    // Allocate memory for the output data structures
    cudaMalloc(&d_outbound, V * sizeof(uint32_t));
    cudaMalloc(&d_inbound, V * sizeof(uint32_t));
    cudaMalloc(&d_weights, V * sizeof(uint32_t));
    cudaMalloc(&d_current_node, sizeof(uint32_t));

    // Transfer inputs to device memory
    cudaMemcpy(d_inbound_vertices, inbound_vertices, E * 2 * sizeof(uint2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outbound_vertices, outbound_vertices, V * sizeof(uint2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, sizeof(uint2), cudaMemcpyHostToDevice);

    // Transfer outputs to device memory
    // TODO: Except for weights (which we initialized to a particular value),
    // are these really needed? Because outbound, inbound are not prepped in
    // cudaRuntime()
    cudaMemcpy(d_outbound, outbound, V * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inbound, inbound, V * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, V * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_current_node, &current_node, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Execute the single-block kernel
    mst << <1, BLOCKSIZE> >> (d_inbound_vertices, d_outbound_vertices, d_shape,
                      d_inbound, d_outbound, d_weights, d_current_node);

    // Copy the results back to host memory
    cudaMemcpy(outbound, d_outbound, V * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(inbound, d_inbound, V * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(weights, d_weights, V * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free.
    cudaFree(d_inbound_vertices);
    cudaFree(d_outbound_vertices);
    cudaFree(d_shape);
    cudaFree(d_inbound);
    cudaFree(d_outbound);
    cudaFree(d_weights);
    cudaFree(d_current_node);
}
