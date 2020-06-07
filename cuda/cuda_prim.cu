#include "cuda_prim.hpp"
//
// CUDA implemenation of Prim's Minimum Spanning Tree Algorithm
//

//////////////////////////
// Options
//////////////////////////

#define NUM_BLOCKS 1
#define BLOCKSIZE 1024


//
// Initialize the compact adjacency list graph representation (Wang et al.)
// 
// See cudaPrimAlgorithm() below for an explanation of this data structure.
//
// vertices must be of length |V|
// edges    must be of length 2*|E|, as each edge will be present twice (once per vertex)
//
void cudaSetup(const Graph& g, uint2 *vertices, uint2 *edges)
{
    uint32_t num_vertices = g.num_vertices();

    // Calculate data for each vertex and the edges to its neighbors 
    for (uint32_t v = 0; v < num_vertices; ++v) {
        std::vector<EdgeTarget> neighbors;
        g.neighbors(v, neighbors);

        // Store neighbor count and offset
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
// Kernel implementing the weight update after an edge has been selected
//
// Uses the compact adjacency list as read-only input, and write the three MST
// data structures. 
//
// See cudaPrimAlgorithm() below for an explanation of this data structure.
//  
__global__ void mst_update(uint2 *vertices, uint2 *edges,
                           uint32_t *outbound, uint32_t *inbound, uint32_t *weights,
                           uint32_t current_vertex, uint32_t update_offset)
{
    // TODO
    // Don't forget: there is a swap operation in here, somewhere (when
    // "moving" a selected edge into the MST)
}


//
// Kernel implementing the min weight determination
//
__global__ void mst_minweight(uint32_t *outbound, uint32_t *inbound, uint32_t *weights /*, what else ?*/)
{
    // TODO
}


//
// Entry point for CUDA Prim's Algorithm
//
// This uses:
//
// Compact Adjacency List   as proposed by Wang et al., based on Harish et al.
// ----------------------
// vertices: for each vertex with number k
//      vertices[k].x = count of immediate neighbors
//      vertices[k].y = offset of edge list in edges
//
// edges: edges ordered by vertex (see above)
//      edges[*].x = vertex on other end
//      edges[*].y = weight of edge
//
//      So for each vertex k:
//         edge list offset = vertices[k].y
//         num_edges        = vertices[k].x
//
//         edges of k       = edges[offset+0]
//                            edges[offset+1]
//                            edges[offset+..]
//                            edges[offset+num_edges]
//
//                            with
//                              edge[...].x = vertex on other end of the edge
//                              edge[...].y = edge weight
//
// => The function cudaSetup() above can be used to initialize such a Compact Adjacency List!
//
//
// MST data structure       as proposed by Wang et al.
// ------------------
// This data structure encodes
//
//      (outbound, inbound) = weight
//
// for the |V|-1 edges of the MST. The edges will be initialized to a ground
// state, and then updated as the algorithm iterates.
//
// Initiallly, the |V|-1 pair are all hypothetically possible edges from the 0
// node to any other node, and with infinite weight to mark the edge as
// non-existent.
//
//      (0, 1)      = inf
//      (0, 2)      = inf
//      ...
//      (0, |V|-1)  = inf
//
void cudaPrimAlgorithm(uint2 *vertices, uint32_t num_vertices, uint2 *edges, uint32_t num_edges,
                       uint32_t *outbound, uint32_t *inbound, uint32_t *weights) {

    // Initialize the MST data structure as per above.
    for (uint32_t i = 0; i < num_vertices - 1; ++i) {
        outbound[i] = 0;
        inbound[i] = i + 1;
        weights[i] = Graph::WEIGHT_INFTY;
    }

    // Data structures in device memory
    uint2 *d_vertices, *d_edges;
    uint32_t *d_outbound, *d_inbound, *d_weights;

    // Allocate memory for the data structures in device memory
    cudaMalloc(&d_vertices, num_vertices * sizeof(uint2));
    cudaMalloc(&d_edges, num_edges * sizeof(uint2));
    cudaMalloc(&d_outbound, num_vertices * sizeof(uint32_t));
    cudaMalloc(&d_inbound, num_vertices * sizeof(uint32_t));
    cudaMalloc(&d_weights, num_vertices * sizeof(uint32_t));
    // Transfer inputs to device memory
    cudaMemcpy(d_vertices, vertices, num_vertices * sizeof(uint2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges, num_edges * sizeof(uint2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outbound, outbound, num_vertices * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inbound, inbound, num_vertices * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, num_vertices * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // The current vertex. We always start with 0
    uint32_t C = 0;

    // Outer loop
    for (uint32_t ii = 0; ii < num_vertices - 1; ++ii) {
        // First, update the weights
        mst_update <<<NUM_BLOCKS, BLOCKSIZE>>> (d_vertices, d_edges,
                                       d_outbound, d_inbound, d_weights,
                                       C, ii);
        // Then, get the node with the minimum weight
        // FIXME Carefull this interface is not yet complete!! I'm still working on it
        //
        // Invoke 1: minimum per block, stored in temporary result
        mst_minweight <<<NUM_BLOCKS, BLOCKSIZE>>> (outbound, inbound, weights /*, what else ?*/);
        // Invoke 2: miminum of temporary result
        mst_minweight <<<1, BLOCKSIZE>>> (outbound, inbound, weights /*, what else ?*/);
        // Update C here
    }

    // Copy the results back to host memory
    cudaMemcpy(outbound, d_outbound, num_vertices * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(inbound, d_inbound, num_vertices * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(weights, d_weights, num_vertices * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_vertices);
    cudaFree(d_edges);
    cudaFree(d_inbound);
    cudaFree(d_outbound);
    cudaFree(d_weights);
}
