#include "cuda_prim.hpp"
#include <cuda_runtime.h>

#define BLOCKSIZE 1024

void cudaSetup(const Graph& g, uint2 *&vertex_adjacent_count_index, uint2 *&edge_target_weight) {
    uint32_t V = g.num_vertices();
    uint32_t E = g.num_edges();
    edge_target_weight = new uint2[E*2];
    vertex_adjacent_count_index = new uint2[V];
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
 * Kernel to initialize the temporary MST data structures.
 */
__global__ void initializeMst(uint32_t *mst_out, uint32_t *mst_in, uint32_t *mst_weight,
                              uint32_t *mst_in_reverse, uint32_t *mst_border,
                              uint32_t *current_node,
                              uint2 *vertex_adjacent_count_index, uint2 *edge_target_weight,
                              uint32_t end)
{
    // find thread working index
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < end) {
        mst_out[idx] = 0;
        mst_in[idx] = idx + 1;
        mst_in_reverse[idx + 1] = idx;
        mst_weight[idx] = UINT32_MAX;
    }
    __syncthreads();
    // now set the weight for the edges going from vertex 0
    if (idx < vertex_adjacent_count_index[0].x) {
        uint32_t pos = idx + vertex_adjacent_count_index[0].y;
        mst_weight[edge_target_weight[pos].x] = edge_target_weight[pos].y;
    }
    if (idx == 0) {
        // set border and start node
        *mst_border = 0;
        *current_node = 0;
    }
}

/**
 * Kernel to find minimum weight edge.
 * It uses the approaches from M. Harris,
 * "Optimizing parallel reduction in CUDA," NVidia, Tech. Rep., 2007.
 * http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * It assumes a weight vector (uint2) with x = key and y = value
 */

/**
 * warp64MinReduce: find the minimum for less than 64 items - with fully unrolled loop.
 */
template <unsigned int blockSize>
__device__ void warp64MinReduce(volatile uint32_t *sKey, volatile uint32_t *sVal, int len, unsigned int tid) {
    if (blockSize >= 64) {
        if (tid+32 < len && sVal[tid+32] < sVal[tid]) {
            sVal[tid] = sVal[tid+32];
            sKey[tid] = sKey[tid+32];
        }
    }
    if (blockSize >= 32) {
        if (tid+16 <= len && sVal[tid+16] < sVal[tid]) {
            sVal[tid] = sVal[tid+16];
            sKey[tid] = sKey[tid+16];
        }
    }
    if (blockSize >= 16) {
        if (tid+8 <= len && sVal[tid+8] < sVal[tid]) {
            sVal[tid] = sVal[tid+8];
            sKey[tid] = sKey[tid+8];
        }
    }
    if (blockSize >= 8) {
        if (tid+4 <= len && sVal[tid+4] < sVal[tid]) {
            sVal[tid] = sVal[tid+4];
            sKey[tid] = sKey[tid+4];
        }
    }
    if (blockSize >= 4) {
        if (tid+2 <= len && sVal[tid+2] < sVal[tid]) {
            sVal[tid] = sVal[tid+2];
            sKey[tid] = sKey[tid+2];
        }
    }
    if (blockSize >= 2) {
        if (tid+1 <= len && sVal[tid+1] < sVal[tid]) {
            sVal[tid] = sVal[tid+1];
            sKey[tid] = sKey[tid+1];
        }
    }
}

__shared__ uint32_t sKey[BLOCKSIZE];
__shared__ uint32_t sVal[BLOCKSIZE];

/**
 * minReduce: step for reduction
 * if blockSize < n this step has to be called a second time.
 * As before the int2 values are read as x = key, y = value.
 * If the parameter inKey is 0 then the key is filled with the index.
 */
template <unsigned int blockSize>
__global__ void minReduce(uint32_t *inKey, uint32_t *inVal, uint32_t *outKey, uint32_t *outVal,
                          uint32_t *start, uint32_t end)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize*2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    int len = end - *start;
    // reduce global memory into shared data
    if (i < *start) {
        i += gridSize*(1+(*start-i-1)/gridSize);
    }
    while (i<end) {
        if (inVal[i] <= inVal[i+blockSize]) {
            sVal[tid] = inVal[i];
            if (inKey == 0) {
                sKey[tid] = i;
            }
            else {
                sKey[tid] = inKey[i];
            }
        }
        else {
            sVal[tid] = inVal[i+blockSize];
            if (inKey == 0) {
                sKey[tid] = i+blockSize;
            }
            else {
                sKey[tid] = inKey[i+blockSize];
            }
        }
        i+= gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256 && tid+256 < len) {
            if (sVal[tid+256] < sVal[tid]) {
                sVal[tid] = sVal[tid+256];
                sKey[tid] = sKey[tid+256];
            }
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128 && tid+128 < len) {
            if (sVal[tid+128] < sVal[tid]) {
                sVal[tid] = sVal[tid+128];
                sKey[tid] = sKey[tid+128];
            }
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64 && tid+64 < len) {
            if (sVal[tid+64] < sVal[tid]) {
                sVal[tid] = sVal[tid+64];
                sKey[tid] = sKey[tid+64];
            }
        }
        __syncthreads();
    }
    if (tid < 32 && tid+1 < len) warp64MinReduce<blockSize>(sKey, sVal, len, tid);
    if (tid == 0) {
        outKey[blockIdx.x] = sKey[0];
        outVal[blockIdx.x] = sVal[0];
    }
}

/**
 * Kernel for doing the move step.
 * If the minimum is not the first edge it is
 * swapped with the first one.
 * This kernel does not need to be parallelized.
 */
__global__ void moveStep(uint32_t *mst_out, uint32_t *mst_in, uint32_t *mst_weight, uint32_t *mst_in_reverse,
                         uint32_t *mst_border, uint32_t *min_index, uint32_t *current_node) {
    // make sure that first in excluded list is the lightest one
    if (*min_index > *mst_border) {
        // swap in ...
        uint32_t swap = mst_in[*mst_border];
        mst_in[*mst_border] = mst_in[*min_index];
        mst_in[*min_index] = swap;
        // ..., out, ...
        swap = mst_out[*mst_border];
        mst_out[*mst_border] = mst_out[*min_index];
        mst_out[*min_index] = swap;
        // ..., weight
        swap = mst_weight[*mst_border];
        mst_weight[*mst_border] = mst_weight[*min_index];
        mst_weight[*min_index] = swap;
        // adjust in_reverse
        mst_in_reverse[mst_in[*mst_border]] = *mst_border;
        mst_in_reverse[mst_in[*min_index]] = *min_index;
    }
    // set new current node
    *current_node = mst_in[*mst_border];
    // increase border between MST and not-yet-MST
    ++(*mst_border);
}

/**
 * Kernel for doing the update step.
 * Each thread will take care of checking one of outbound
 * edges going from the current node to see whether the target of
 * it has a higher weight in the temporary structures
 */
__global__ void updateStep(uint32_t *mst_out, uint32_t *mst_in, uint32_t *mst_weight,
                           uint32_t *mst_in_reverse, uint32_t *mst_border,
                           uint32_t *current_node,
                           uint2 *vertex_adjacent_count_index, uint2 *edge_target_weight)
{
    // find thread working index
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    // find edges going from current node
    uint2 current_count_index = vertex_adjacent_count_index[*current_node];
    if (idx < current_count_index.x) {
        uint2 cur_target_weight = edge_target_weight[current_count_index.y+idx];
        uint32_t in_idx = mst_in_reverse[cur_target_weight.x];
        if (cur_target_weight.y < mst_weight[in_idx]) {
            // update mst_out and mst_weight
            mst_out[in_idx] = *current_node;
            mst_weight[in_idx] = cur_target_weight.y;
        }
    }
}

void cudaPrimAlgorithm(uint2 *vertex_adjacent_count_index, uint2 *edge_target_weight,
                       uint32_t *mst_out, uint32_t *mst_in, uint32_t *mst_weight,
                       uint32_t V, uint32_t E)
{
    // copy graph info into device memory
    uint2 *d_vertex_adjacent_count_index;
    cudaMalloc(&d_vertex_adjacent_count_index, V*sizeof(uint2));
    cudaMemcpy(d_vertex_adjacent_count_index, vertex_adjacent_count_index,
               V*sizeof(uint2), cudaMemcpyHostToDevice);
    uint2 *d_edge_target_weight;
    cudaMalloc(&d_edge_target_weight, 2*E*sizeof(uint2));
    cudaMemcpy(d_edge_target_weight, edge_target_weight,
               2*E*sizeof(uint2), cudaMemcpyHostToDevice);
    // number of blocks needed
    uint32_t num_blocks = 1 + (V/BLOCKSIZE);    
    // allocate memory for the MST data
    uint32_t *d_mst_out;
    uint32_t *d_mst_in;
    uint32_t *d_mst_in_reverse;
    uint32_t *d_mst_weight;
    cudaMalloc(&d_mst_out, V*sizeof(uint32_t));
    cudaMalloc(&d_mst_in, V*sizeof(uint32_t));
    cudaMalloc(&d_mst_in_reverse, V*sizeof(uint32_t));
    cudaMalloc(&d_mst_weight, V*sizeof(uint32_t));
    // allocate memory for the current node & border
    uint32_t *d_mst_border;
    uint32_t *d_current_node;
    cudaMalloc(&d_mst_border, sizeof(uint32_t));
    cudaMalloc(&d_current_node, sizeof(uint32_t));
    // allocate memory for temp space when doing minReduce
    uint32_t *d_min_key_array;
    uint32_t *d_min_val_array;
    cudaMalloc(&d_min_key_array, V*sizeof(uint32_t));
    cudaMalloc(&d_min_val_array, V*sizeof(uint32_t));
    // allocate memory for minimum found
    uint32_t *d_min_key;
    uint32_t *d_min_val;
    if (num_blocks > 1) {
        cudaMalloc(&d_min_key, sizeof(uint32_t));
        cudaMalloc(&d_min_val, sizeof(uint32_t));
    }
    else {
        d_min_key = d_min_key_array;
        d_min_val = d_min_val_array;
    }

    // initialize MST data structure
    initializeMst <<<num_blocks,BLOCKSIZE>>> (
        d_mst_out, d_mst_in, d_mst_weight, d_mst_in_reverse, d_mst_border,
        d_current_node, d_vertex_adjacent_count_index, d_edge_target_weight,
        V);
    // loop: create MST with V nodes (one is already there)
    for (uint32_t i=1; i<V; i++) {
        // find step: two minReduce steps
        minReduce<BLOCKSIZE> <<<num_blocks,BLOCKSIZE>>> (
            (uint32_t *)0, d_mst_weight, d_min_key_array, d_min_val_array,
            d_mst_border, V);
        if (num_blocks > 1) {
            minReduce<BLOCKSIZE> <<<1,num_blocks>>> (
                d_min_key_array, d_min_val_array, d_min_key, d_min_val,
                d_mst_border, num_blocks);
        }
        // move step: swap if needed, advance indices
        moveStep <<<1, 1>>> (
            d_mst_out, d_mst_in, d_mst_weight, d_mst_in_reverse,
            d_mst_border, d_min_key, d_current_node);
        // update step: look for weights to be updated
        updateStep <<<num_blocks,BLOCKSIZE>>> (
            d_mst_out, d_mst_in, d_mst_weight,
            d_mst_in_reverse, d_mst_border,
            d_current_node,
            d_vertex_adjacent_count_index, d_edge_target_weight);
    }

    // copy memory back to caller
    cudaMemcpy(mst_out, d_mst_out, V*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(mst_in, d_mst_in, V*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(mst_weight, d_mst_weight, V*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // free all memory
    cudaFree(d_vertex_adjacent_count_index);
    cudaFree(d_edge_target_weight);
    cudaFree(d_mst_out);
    cudaFree(d_mst_in);
    cudaFree(d_mst_in_reverse);
    cudaFree(d_mst_weight);
    cudaFree(d_mst_border);
    cudaFree(d_current_node);
    cudaFree(d_min_key_array);
    cudaFree(d_min_val_array);
    if (num_blocks > 1) {
        cudaFree(d_min_key);
        cudaFree(d_min_val);
    }
}
