#include "thrust_prim.hpp"
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cstdint>

#define BLOCKSIZE 1024

void thrustSetup(const Graph& g, thrust::host_vector<uint2> &vertex_adjacent_count_index, thrust::host_vector<uint2> &edge_target_weight) {
    uint32_t V = g.num_vertices();
    uint32_t E = g.num_edges();
    uint32_t pos = 0;
    for (uint32_t v = 0; v < V; ++v) {
        std::vector<EdgeTarget> neighbors;
        g.neighbors(v, neighbors);
        vertex_adjacent_count_index[v].x = neighbors.size();
        vertex_adjacent_count_index[v].y = pos;
        for (auto nb = neighbors.begin(); nb < neighbors.end(); ++nb) {
            edge_target_weight[pos].x = nb->vertex_to;
            edge_target_weight[pos++].y = nb->weight;
        }
    }
}

/**
 * Kernel for doing the move step.
 * If the minimum is not the first edge it is
 * swapped with the first one.
 * This kernel does not need to be parallelized.
 */
__global__ void moveStep2(uint32_t *mst_out, uint32_t *mst_in, uint32_t *mst_weight, uint32_t *mst_in_reverse,
                         uint32_t mst_border, uint32_t min_index) {
    // make sure that first in excluded list is the lightest one
    if (min_index > mst_border) {
        // swap in ...
        uint32_t swap = mst_in[mst_border];
        mst_in[mst_border] = mst_in[min_index];
        mst_in[min_index] = swap;
        // ..., out, ...
        swap = mst_out[mst_border];
        mst_out[mst_border] = mst_out[min_index];
        mst_out[min_index] = swap;
        // ..., weight
        swap = mst_weight[mst_border];
        mst_weight[mst_border] = mst_weight[min_index];
        mst_weight[min_index] = swap;
        // adjust in_reverse
        mst_in_reverse[mst_in[mst_border]] = mst_border;
        mst_in_reverse[mst_in[min_index]] = min_index;
    }
}

/**
 * Kernel for doing the update step.
 * Each thread will take care of checking one of outbound
 * edges going from the current node to see whether the target of
 * it has a higher weight in the temporary structures
 */
__global__ void updateStep2(uint32_t *mst_out, uint32_t *mst_in, uint32_t *mst_weight,
                           uint32_t *mst_in_reverse, uint32_t mst_border,
                           uint2 *vertex_adjacent_count_index, uint2 *edge_target_weight)
{
    // find thread working index
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    // find current node
    uint32_t current_node = 0;
    if (mst_border > 0) {
        current_node = mst_in[mst_border-1];
    }
    // find edges going from current node
    uint2 current_count_index = vertex_adjacent_count_index[current_node];
    if (idx < current_count_index.x) {
        uint2 cur_target_weight = edge_target_weight[current_count_index.y+idx];
        uint32_t in_idx = mst_in_reverse[cur_target_weight.x];
        if (cur_target_weight.y < mst_weight[in_idx] && cur_target_weight.x >= mst_border) {
            // update mst_out and mst_weight
            mst_out[in_idx] = current_node;
            mst_weight[in_idx] = cur_target_weight.y;
        }
    }
}

void thrustPrimAlgorithm(const thrust::host_vector<uint2> &vertex_adjacent_count_index, const thrust::host_vector<uint2> &edge_target_weight,
                         thrust::host_vector<uint32_t> &mst_out, thrust::host_vector<uint32_t> &mst_in,
                         thrust::host_vector<uint32_t> &mst_weight,
                         uint32_t V, uint32_t E)
{
    // copy graph info into device memory
    thrust::device_vector<uint2> d_vertex_adjacent_count_index = vertex_adjacent_count_index;
    thrust::device_vector<uint2> d_edge_target_weight = edge_target_weight;
    // number of blocks needed
    uint32_t num_blocks = 1 + (V/BLOCKSIZE);
    // allocate memory for the MST data
    thrust::device_vector<uint32_t> d_mst_out(V, 0);
    thrust::device_vector<uint32_t> d_mst_weight(V, UINT32_MAX);
    thrust::device_vector<uint32_t> d_mst_in(V);
    thrust::device_vector<uint32_t> d_mst_in_reverse(V);
    thrust::counting_iterator<int> it_0(0);
    thrust::counting_iterator<int> it_1(1);
    thrust::counting_iterator<int> it_V(V);
    thrust::counting_iterator<int> it_V_1(V-1);
    // initialize MST in and in-reverse pointers
    // my attempts to use thrust::sequence raised an exception
    thrust::copy(it_1, it_V, d_mst_in.begin());
    thrust::copy(it_0, it_V_1, d_mst_in_reverse.begin()+1);

    // raw versions of the pointers
    uint2 *raw_vertex_adjacent_count_index = thrust::raw_pointer_cast(&d_vertex_adjacent_count_index[0]);
    uint2 *raw_edge_target_weight = thrust::raw_pointer_cast(&d_edge_target_weight[0]);
    uint32_t *raw_mst_out = thrust::raw_pointer_cast(&d_mst_out[0]);
    uint32_t *raw_mst_weight = thrust::raw_pointer_cast(&d_mst_weight[0]);
    uint32_t *raw_mst_in = thrust::raw_pointer_cast(&d_mst_in[0]);
    uint32_t *raw_mst_in_reverse = thrust::raw_pointer_cast(&d_mst_in_reverse[0]);

    // update first round - using CUDA kernel
    updateStep2<<<num_blocks,BLOCKSIZE>>>(
        raw_mst_out, raw_mst_in, raw_mst_weight,
        raw_mst_in_reverse, 0,
        raw_vertex_adjacent_count_index, raw_edge_target_weight);
    // loop: create MST with V nodes (one is already there)
    for (uint32_t i=0; i<V-1;) {
        // find minimum weight after position i
        auto minPos = thrust::min_element(d_mst_weight.begin()+i, d_mst_weight.end());
        int minIdx = minPos - d_mst_weight.begin();
        // swap element at border with minimal element
        moveStep2<<<1,1>>>(raw_mst_out, raw_mst_in, raw_mst_weight, raw_mst_in_reverse,
            i, minIdx);
        // update step
        ++i;
        updateStep2<<<num_blocks,BLOCKSIZE>>>(
            raw_mst_out, raw_mst_in, raw_mst_weight,
            raw_mst_in_reverse, i,
            raw_vertex_adjacent_count_index, raw_edge_target_weight);
    }
    // copy memory back to caller
    thrust::copy(d_mst_out.begin(), d_mst_out.end(), mst_out.begin());
    thrust::copy(d_mst_in.begin(), d_mst_in.end(), mst_in.begin());
    thrust::copy(d_mst_weight.begin(), d_mst_weight.end(), mst_weight.begin());
    // memory will be cleared automatically
}
