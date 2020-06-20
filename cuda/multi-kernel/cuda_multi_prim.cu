//
// Second CUDA implemenation of Prim's Minimum Spanning Tree Algorithm
//
//
// Please refer to the report for documentation on all the data structures used
// here, as well as an outline of the implementation.

#include "cuda_multi_prim.hpp"

//
// Kernel implementing the first phase of min reduction primitive
//
// local block minima are stored in a temporary array v_red
//
__global__ void min_reduction1(uint32_t *inbound, uint32_t *weights, uint2 *v_red, uint32_t num_vertices) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	extern __shared__ uint2 shm[];

	// Initial assignment of shared memory -> of node is note reachable the weight is set to +inf
	shm[threadIdx.x].x = idx;
	shm[threadIdx.x + blockDim.x].x = idx + blockDim.x;
	shm[threadIdx.x].y = idx < num_vertices && inbound[idx] > num_vertices ? weights[idx] : UINT32_MAX;
	shm[threadIdx.x + blockDim.x].y = UINT32_MAX;

	// reduction loop
	for (int j = blockDim.x * SHM_FACTOR; j > 1; j /= 2) {
		for (int k = 0; k < SHM_FACTOR; k++) {
			if (shm[threadIdx.x].y > shm[threadIdx.x + j / 2].y) {
				shm[threadIdx.x].x = shm[threadIdx.x + j / 2].x;
				shm[threadIdx.x].y = shm[threadIdx.x + j / 2].y;
			}
		}
		__syncthreads();
	}
	

	// store best local solution in temporary array
	if (threadIdx.x == 0) {
		v_red[blockIdx.x].x = shm[0].x;
		v_red[blockIdx.x].y = shm[0].y;
	}
}

//
// Kernel implementing the second phase of min reduction primitive
//
// temporary reduction array v_red is reduced and best solution stored in v_red[0]
//
__global__ void min_reduction2(uint2 *v_red, uint32_t *current_node, uint32_t *last_node, uint32_t red1_blocks) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	// if only one block available -> best solution is already in v_red[0]
	if (red1_blocks == 1) {
		if (idx == 1) {
			*last_node = *current_node;
			*current_node = v_red[0].x;
		}
		return;
	}

	uint32_t half_size = red1_blocks / 2;

	// reduction loop
	for (int j = half_size; j > 1; j /= 2) {
		for (int i = 0; i < j; i += blockDim.x) {
			if (idx + i < j) {
				if (v_red[idx + i + j].y < v_red[idx + i].y) {
					v_red[idx + i].x = v_red[idx + i + j].x;
					v_red[idx + i].y = v_red[idx + i + j].y;
				}
			}
		}
		__syncthreads();
	}
	// adjust current and last nodes to best result
	if (idx == 0) {
		*last_node = *current_node;
		if (v_red[1].y < v_red[0].y) {
			*current_node = v_red[1].x;
		}
		else {
			*current_node = v_red[0].x;
		}
	}
}

//
// Kernel implementing the first weight update phase primitive
//
// Uses the compact adjacency list as read-only input, and writes to the MST
// data structure.
//
// Each thread accesses only one "row" of the MST data structure, so there is
// no need to synchronize anything.
//
// The position in the solution array is the corresponding inbound node of the new nodes reachable from current_node
//  
__global__ void update_mst(uint2 *outbound_vertices, uint2 *inbound_vertices, uint32_t *outbound, uint32_t *weights, uint32_t *current_node) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	uint32_t start_index = outbound_vertices[*current_node].y;
	uint32_t end_index = start_index + outbound_vertices[*current_node].x;

	if (idx < end_index - start_index) {
		uint32_t edge_idx = idx + start_index;
		if (inbound_vertices[edge_idx].y < weights[inbound_vertices[edge_idx].x]) {
			weights[inbound_vertices[edge_idx].x] = inbound_vertices[edge_idx].y;
			outbound[inbound_vertices[edge_idx].x] = *current_node;
		}
	}
}

//
// Kernel implementing the second weight update phase primitive
//
// Take the best fitting edge and store it store it at x-th position in the solution array. x is the last node found
//  
__global__ void update_mst2(uint32_t *outbound, uint32_t *inbound, uint32_t *weights, uint32_t *current_node, uint32_t *last_node) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx == 0) {
		outbound[*last_node] = outbound[*current_node];
		inbound[*last_node] = *current_node;
		weights[*last_node] = weights[*current_node];
		weights[*current_node] = UINT32_MAX;
	}
}

//
// Initialize the compact adjacency list representation (Wang et al.)
//
// Refer to the report for a detailed explanation of this data structure.
//
// The input graph is generated using our own graph generator, which can be
// found in base/.
//
void cuda_multi_setup(const Graph& g, uint2 *&inbound_vertices, uint2 *&outbound_vertices) {
	uint32_t pos = 0;
	for (uint32_t v = 0; v < g.num_vertices(); ++v) {
		std::vector<EdgeTarget> neighbors;
		g.neighbors(v, neighbors);
		outbound_vertices[v].x = neighbors.size();
		outbound_vertices[v].y = v == 0 ? 0 : v == 1 ? outbound_vertices[v - 1].x : outbound_vertices[v - 1].y + outbound_vertices[v - 1].x;
		for (auto nb = neighbors.begin(); nb < neighbors.end(); ++nb) {
			inbound_vertices[pos].x = nb->vertex_to;
			inbound_vertices[pos++].y = nb->weight;
		}
	}
}

// allocates all resources needed on the device
void allocate_resources(uint32_t num_vertices, uint32_t num_edges, uint2 *& inbound_vertices, uint2 *& outbound_vertices, uint2 *& d_inbound_vertices, uint2 *& d_outbound_vertices, uint2 *&d_red_array, uint32_t *outbound, uint32_t *inbound, uint32_t *weights, uint32_t current_node, uint32_t *&d_outbound, uint32_t *&d_inbound, uint32_t *&d_weights, uint32_t *&d_current_node, uint32_t *&d_last_node, uint32_t num_blocks) {
	cudaMalloc(&d_inbound_vertices, num_edges * 2 * sizeof(uint2));
	cudaMalloc(&d_outbound_vertices, num_vertices * sizeof(uint2));

	cudaMalloc(&d_outbound, num_vertices * sizeof(uint32_t));
	cudaMalloc(&d_inbound, num_vertices * sizeof(uint32_t));
	cudaMalloc(&d_weights, num_vertices * sizeof(uint32_t));
	cudaMalloc(&d_current_node, sizeof(uint32_t));
	cudaMalloc(&d_last_node, sizeof(uint32_t));

	cudaMalloc(&d_red_array, num_blocks * sizeof(uint2));

	cudaMemcpy(d_inbound_vertices, inbound_vertices, num_edges * 2 * sizeof(uint2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_outbound_vertices, outbound_vertices, num_vertices * sizeof(uint2), cudaMemcpyHostToDevice);

	cudaMemcpy(d_outbound, outbound, num_vertices * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inbound, inbound, num_vertices * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights, num_vertices * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_current_node, &current_node, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

// frees all allocated resources on the device
void free_resources(uint2 *& d_inbound_vertices, uint2 *& d_outbound_vertices, uint2 *&d_red_array, uint32_t *&d_outbound, uint32_t *&d_inbound, uint32_t *&d_weights, uint32_t *&d_current_node, uint32_t *&d_last_node) {
	cudaFree(d_inbound_vertices);
	cudaFree(d_outbound_vertices);

	cudaFree(d_inbound);
	cudaFree(d_outbound);
	cudaFree(d_weights);
	cudaFree(d_current_node);
	cudaFree(d_last_node);
	cudaFree(d_red_array);
}

// function for calculating an optimal number of threads to the current vertices count
uint32_t get_num_threads(uint32_t num_vertices) {
	if (num_vertices < 8196) {
		return 32;
	}
	else if (num_vertices < 16384) {
		return 128;
	}
	else if(num_vertices < 131072) {
		return 512;
	}
	else {
		return 1024;
	}
}
// function for calculating an optimal number of blocks to the current vertices and thread count
uint32_t get_num_blocks(uint32_t num_vertices, uint32_t num_threads) {
	uint32_t blockfactor = (num_vertices - 1) / num_threads;
	uint32_t num_blocks = 1;
	while (blockfactor != 0) {
		blockfactor = blockfactor >> 1;
		num_blocks = num_blocks << 1;
	}
	return num_blocks;
}

void cuda_multi_prim_algorithm(uint32_t num_vertices, uint32_t num_edges, uint2 *outbound_vertices, uint2 *inbound_vertices, uint32_t *outbound, uint32_t *inbound, uint32_t *weights) {
	{
		// declaration of device pointers
		uint2 * d_inbound_vertices = NULL, *d_outbound_vertices = NULL;
		uint32_t *d_outbound = NULL, *d_inbound = NULL, *d_weights = NULL;
		uint2 *d_red_array = NULL;

		// start node
		uint32_t current_node = 0, *d_current_node = 0, *d_last_node = NULL;

		// calculate an optimal distribution od blocks and threads
		uint32_t num_threads = get_num_threads(num_vertices);
		uint32_t num_blocks = get_num_blocks(num_vertices, num_threads);

		// allocate resources
		allocate_resources(num_vertices, num_edges,inbound_vertices, outbound_vertices, d_inbound_vertices, d_outbound_vertices, d_red_array, outbound, inbound, weights, current_node, d_outbound, d_inbound, d_weights, d_current_node, d_last_node, num_blocks);

		// calculate the size of the shared memory needed on the device. This value is given to the kernel as third parameter
		uint32_t shm_size = num_threads * sizeof(uint2) * SHM_FACTOR;

		// main loop where prim's algorithm is executed
		for (int i = 0; i < num_vertices - 1; i++) {
			update_mst << <num_blocks, num_threads >> > (d_outbound_vertices, d_inbound_vertices, d_outbound, d_weights, d_current_node);
			min_reduction1 << <num_blocks, num_threads, shm_size >> > (d_inbound, d_weights, d_red_array, num_vertices);
			min_reduction2 << <1, num_threads >> > (d_red_array, d_current_node, d_last_node, num_blocks);
			update_mst2 << <num_blocks, num_threads >> > (d_outbound, d_inbound, d_weights, d_current_node, d_last_node);
		}

		// copy results from device
		cudaMemcpy(outbound, d_outbound, num_vertices * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(inbound, d_inbound, num_vertices * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(weights, d_weights, num_vertices * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		// free resources
		free_resources(d_inbound_vertices, d_outbound_vertices, d_red_array, d_outbound, d_inbound, d_weights, d_current_node, d_last_node);
	}
}