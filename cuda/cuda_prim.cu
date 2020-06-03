#include "cuda_prim.hpp"

#define BLOCKSIZE 1024

void cudaSetup(const Graph& g, uint2 *&inbound_vertices, uint2 *&outbound_vertices, uint2 *&shape) {
    shape = new uint2;
    shape->x = g.num_vertices();
    shape->y = g.num_edges();
    inbound_vertices = new uint2[shape->y * 2];
    outbound_vertices = new uint2[shape->x];
    uint32_t pos = 0;
    for (uint32_t v = 0; v < shape->x; ++v) {
        std::vector<EdgeTarget> neighbors;
        g.neighbors(v, neighbors);
        outbound_vertices[v].x = neighbors.size();
        outbound_vertices[v].y = v == 0 ? 0 : v == 1 ? outbound_vertices[v-1].x : outbound_vertices[v - 1].y + outbound_vertices[v-1].x;
        for (auto nb = neighbors.begin(); nb < neighbors.end(); ++nb) {
            inbound_vertices[pos].x = nb->vertex_to;
            inbound_vertices[pos++].y = nb->weight;
        }
    }
}

__global__ void mst(uint2 *inbound_vertices, uint2 *outbound_vertices, uint2 *shape,
                    uint32_t *inbound, uint32_t *outbound, uint32_t *weights, uint32_t *current_node)
{
    __shared__ uint2 shm[BLOCKSIZE];

    if (threadIdx.x < BLOCKSIZE) {
        shm[threadIdx.x].y = UINT32_MAX;
    }

    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int max_nodes = shape->x;

    for (int i = 0; i < max_nodes - 1; i++) {

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

void cudaPrimAlgorithm(uint2 *inbound_vertices, uint2 *outbound_vertices, uint2 *shape,
    uint32_t *inbound, uint32_t *outbound, uint32_t *weights) {
const uint32_t V = shape->x;
const uint32_t E = shape->y;
uint2 * d_inbound_vertices, *d_outbound_vertices, *d_shape;
uint32_t *d_outbound, *d_inbound, *d_weights;
uint32_t current_node = 0, *d_current_node;

std::fill_n(weights, V, UINT32_MAX);
cudaMalloc(&d_inbound_vertices, E * 2 * sizeof(uint2));
cudaMalloc(&d_outbound_vertices, V * sizeof(uint2));
cudaMalloc(&d_shape, sizeof(uint2));

cudaMalloc(&d_outbound, V * sizeof(uint32_t));
cudaMalloc(&d_inbound, V * sizeof(uint32_t));
cudaMalloc(&d_weights, V * sizeof(uint32_t));
cudaMalloc(&d_current_node, sizeof(uint32_t));

cudaMemcpy(d_inbound_vertices, inbound_vertices, E * 2 * sizeof(uint2), cudaMemcpyHostToDevice);
cudaMemcpy(d_outbound_vertices, outbound_vertices, V * sizeof(uint2), cudaMemcpyHostToDevice);
cudaMemcpy(d_shape, shape, sizeof(uint2), cudaMemcpyHostToDevice);

cudaMemcpy(d_outbound, outbound, V * sizeof(uint32_t), cudaMemcpyHostToDevice);
cudaMemcpy(d_inbound, inbound, V * sizeof(uint32_t), cudaMemcpyHostToDevice);
cudaMemcpy(d_weights, weights, V * sizeof(uint32_t), cudaMemcpyHostToDevice);
cudaMemcpy(d_current_node, &current_node, sizeof(uint32_t), cudaMemcpyHostToDevice);

mst << <1, BLOCKSIZE >> > (d_inbound_vertices, d_outbound_vertices, d_shape, d_inbound, d_outbound, d_weights, d_current_node);

cudaMemcpy(outbound, d_outbound, V * sizeof(uint32_t), cudaMemcpyDeviceToHost);
cudaMemcpy(inbound, d_inbound, V * sizeof(uint32_t), cudaMemcpyDeviceToHost);
cudaMemcpy(weights, d_weights, V * sizeof(uint32_t), cudaMemcpyDeviceToHost);

cudaFree(d_inbound_vertices);
cudaFree(d_outbound_vertices);
cudaFree(d_shape);

cudaFree(d_inbound);
cudaFree(d_outbound);
cudaFree(d_weights);
cudaFree(d_current_node);
}
