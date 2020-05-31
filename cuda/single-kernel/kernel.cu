#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "matrix_graph.hpp"
#include "graph.hpp"

#define BLOCKSIZE 1024

__device__ void print(uint2 * array, uint32_t idx, unsigned int size) {
	if (idx == 1) {
		printf("%u:\n", size);
		for (uint32_t i = 0; i < size; i++) {
			printf("%u %u\n", array[i].y, array[i].x);
		}
		printf("\n");
	}
}

__device__ void print(uint3 * array, uint32_t idx, unsigned int size) {
	if (idx == 1) {
		printf("%d:\n", size);
		for (uint32_t i = 0; i < size; i++) {
			printf("%u %u %u\n", array[i].x, array[i].y, array[i].z);
		}
		printf("\n");
	}
}

__device__ void print(uint32_t * outbound, uint32_t * inbound, uint32_t * weights, uint32_t idx, unsigned int size) {
	if (idx == 1) {
		printf("%u:\n", size);
		for (uint32_t i = 0; i < size; i++) {
			printf("%u -> %u: weight %u\n", outbound[i], inbound[i], weights[i]);
		}
		printf("\n");
	}
}

__global__ void mst(uint2 *inbound_vertices, uint2 *outbound_vertices, uint2 *shape, uint32_t *inbound, uint32_t *outbound, uint32_t *weights, uint32_t *current_node)
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
			//printf("END LOOP:\n");
		}
		__syncthreads();
	}
}




