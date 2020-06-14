#include <iostream>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include "list_graph.hpp"
#include "generator.hpp"
#include "graph.hpp"
#include "print_helper.cu"

#include <chrono>


#define NUM_RUNS 1
#define SHM_FACTOR 2

#define NUM_VERTICES 513
#define DENSITY 0.2
#define MIN_WEIGHT 0
#define MAX_WEIGHT 50

using namespace std;

__global__ void min_reduction1(uint32_t *inbound, uint32_t *weights, uint2 *v_red) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	extern __shared__ uint2 shm[];

	shm[threadIdx.x].x = idx;
	shm[threadIdx.x + blockDim.x].x = idx + blockDim.x;
	shm[threadIdx.x].y = idx < NUM_VERTICES && inbound[idx] > NUM_VERTICES ? weights[idx] : UINT32_MAX;
	shm[threadIdx.x + blockDim.x].y = UINT32_MAX;
	
	__syncthreads();

	for (int j = blockDim.x * SHM_FACTOR; j > 1; j /= 2) {
		for (int k = 0; k < SHM_FACTOR; k++) {
			if (shm[threadIdx.x].y > shm[threadIdx.x + j / 2].y) {
				shm[threadIdx.x].x = shm[threadIdx.x + j / 2].x;
				shm[threadIdx.x].y = shm[threadIdx.x + j / 2].y;
			}
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		v_red[blockIdx.x].x = shm[0].x;
		v_red[blockIdx.x].y = shm[0].y;
	}
}

__global__ void min_reduction2(uint2 *v_red, uint32_t *current_node, uint32_t *last_node, uint32_t red1_blocks) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (red1_blocks == 1) {
		if (idx == 1) {
			*last_node = *current_node;
			*current_node = v_red[0].x;
		}
		return;
	}

	uint32_t half_size = red1_blocks / 2;

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

__global__ void update_mst2(uint32_t *outbound, uint32_t *inbound, uint32_t *weights, uint32_t *current_node, uint32_t *last_node) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx == 0) {
		outbound[*last_node] = outbound[*current_node];
		inbound[*last_node] = *current_node;
		weights[*last_node] = weights[*current_node];
		weights[*current_node] = UINT32_MAX;
	}
}


void cuda_setup(const Graph& g, uint2 *&inbound_vertices, uint2 *&outbound_vertices, uint2 *&shape) {
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
		outbound_vertices[v].y = v == 0 ? 0 : v == 1 ? outbound_vertices[v - 1].x : outbound_vertices[v - 1].y + outbound_vertices[v - 1].x;
		for (auto nb = neighbors.begin(); nb < neighbors.end(); ++nb) {
			inbound_vertices[pos].x = nb->vertex_to;
			inbound_vertices[pos++].y = nb->weight;
		}
	}
}

void allocate_resources(uint2 *& inbound_vertices, uint2 *& outbound_vertices, uint2 *&shape, uint2 *& d_inbound_vertices, uint2 *& d_outbound_vertices, uint2 *&d_shape, uint2 *&d_red_array, uint32_t *outbound, uint32_t *inbound, uint32_t *weights, uint32_t current_node, uint32_t *&d_outbound, uint32_t *&d_inbound, uint32_t *&d_weights, uint32_t *&d_current_node, uint32_t *&d_last_node, uint32_t num_blocks) {
	cudaMalloc(&d_inbound_vertices, shape->y * 2 * sizeof(uint2));
	cudaMalloc(&d_outbound_vertices, shape->x * sizeof(uint2));
	cudaMalloc(&d_shape, sizeof(uint2));

	cudaMalloc(&d_outbound, shape->x * sizeof(uint32_t));
	cudaMalloc(&d_inbound, shape->x * sizeof(uint32_t));
	cudaMalloc(&d_weights, shape->x * sizeof(uint32_t));
	cudaMalloc(&d_current_node, sizeof(uint32_t));
	cudaMalloc(&d_last_node, sizeof(uint32_t));

	cudaMalloc(&d_red_array, num_blocks * sizeof(uint2));

	cudaMemcpy(d_inbound_vertices, inbound_vertices, shape->y * 2 * sizeof(uint2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_outbound_vertices, outbound_vertices, shape->x * sizeof(uint2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_shape, shape, sizeof(uint2), cudaMemcpyHostToDevice);

	cudaMemcpy(d_outbound, outbound, shape->x * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inbound, inbound, shape->x * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights, shape->x * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_current_node, &current_node, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void free_resources(uint2 *& inbound_vertices, uint2 *& outbound_vertices, uint2 *&shape, uint2 *& d_inbound_vertices, uint2 *& d_outbound_vertices, uint2 *&d_shape, uint2 *&d_red_array, uint32_t *outbound, uint32_t *inbound, uint32_t *weights, uint32_t *&d_outbound, uint32_t *&d_inbound, uint32_t *&d_weights, uint32_t *&d_current_node, uint32_t *&d_last_node) {
	cudaFree(d_inbound_vertices);
	cudaFree(d_outbound_vertices);
	cudaFree(d_shape);

	cudaFree(d_inbound);
	cudaFree(d_outbound);
	cudaFree(d_weights);
	cudaFree(d_current_node);
	cudaFree(d_last_node);

	cudaFree(d_red_array);

	delete[] inbound_vertices;
	delete[] outbound_vertices;
	delete[] shape;
}

uint32_t calc_num_blocks(uint32_t num_vertices) {
	if (num_vertices < 512) {
		return 0;
	}
	uint32_t sqr = sqrt(num_vertices);
	uint32_t factor = 1;
	while (sqr != 0) {
		sqr = sqr >> 1;
		factor++;
	}
	return factor;
}

void print_result(uint32_t * outbound, uint32_t *inbound, uint32_t *weights, uint32_t V) {
	cout << "H " << V << " " << V - 1 << " " << 1 << endl;
	uint32_t counter = 0;
	for (int i = 0; i < NUM_VERTICES; i++) {
		if (inbound[i] < NUM_VERTICES) {
			cout << "E " << outbound[i] << " " << inbound[i] << " " << weights[i] << endl;
			counter++;
		}
	}
	cout << "NUMBER LINES : " << counter << endl;
}

void print_raw(uint32_t * outbound, uint32_t *inbound, uint32_t *weights) {
	cout << "H " << NUM_VERTICES << " " << NUM_VERTICES << " " << 1 << endl;
	uint32_t counter = 0;
	for (int i = 0; i < NUM_VERTICES; i++) {
			cout << "E " << outbound[i] << " " << inbound[i] << " " << weights[i] << endl;
	}
}

int main()
{
	ListGraph g;

	chrono::steady_clock::time_point begin, end;
	double runtime;

	uint2 * inbound_vertices, *outbound_vertices, *shape = NULL;
	uint2 * d_inbound_vertices = NULL, *d_outbound_vertices = NULL, *d_shape = NULL;

	uint32_t *d_outbound = NULL, *d_inbound = NULL, *d_weights = NULL;

	generator(g, NUM_VERTICES, MIN_WEIGHT, MAX_WEIGHT, DENSITY, false, UINT64_MAX);
	//cin >> g;

	// write to stdout
	//cout << g << endl;

	for (int i = 0; i < NUM_RUNS; i++) {

		cuda_setup(g, inbound_vertices, outbound_vertices, shape);

		uint32_t outbound[NUM_VERTICES];
		uint32_t inbound[NUM_VERTICES];
		uint32_t weights[NUM_VERTICES];
		uint2 *d_red_array = NULL;

		fill(weights, weights + NUM_VERTICES, UINT32_MAX);

		// start node
		uint32_t current_node = 0, *d_current_node = NULL, *d_last_node = NULL;

		/*
		cout << "outbound:" << endl;
		for (int i = 0; i < shape->x; i++) {
			printf("%d %d\n", outbound_vertices[i].y, outbound_vertices[i].x);
		}
		cout << "inbound:" << endl;
		for (int i = 0; i < shape->y * 2; i++) {
			printf("%d %d\n", inbound_vertices[i].y, inbound_vertices[i].x);
		}
		cout << "shape:" << endl;
		cout << "Number of Vertices: " << shape[0].x << endl << "Number of edges: " << shape[0].y << endl;
		*/

		uint32_t num_blocks_factor = calc_num_blocks(NUM_VERTICES);

		uint32_t num_blocks = 1 << num_blocks_factor;
		uint32_t num_threads = num_blocks_factor == 0 ? NUM_VERTICES : 1 << (num_blocks_factor - 2);

		allocate_resources(inbound_vertices, outbound_vertices, shape, d_inbound_vertices, d_outbound_vertices, d_shape, d_red_array, outbound, inbound, weights, current_node, d_outbound, d_inbound, d_weights, d_current_node, d_last_node, num_blocks);

		cout << "NUM BLOCKS " << num_blocks << "NUM THREADS " << num_threads << endl;
		uint32_t shm_size = num_threads * sizeof(uint2) * SHM_FACTOR;

		begin = chrono::steady_clock::now();
		for (int i = 0; i < NUM_VERTICES - 1; i++) {
			update_mst << <num_blocks, num_threads >> > (d_outbound_vertices, d_inbound_vertices, d_outbound, d_weights, d_current_node);
			min_reduction1 << <num_blocks, num_threads, shm_size >> > (d_inbound, d_weights, d_red_array);
			min_reduction2 << <1, num_threads >> > (d_red_array, d_current_node, d_last_node, num_blocks);
			update_mst2 << <num_blocks, num_threads >> > (d_outbound, d_inbound, d_weights, d_current_node, d_last_node);
		}
		end = chrono::steady_clock::now();
		runtime = (chrono::duration_cast<chrono::duration<double>>(end - begin)).count() * 1000;

		cudaMemcpy(outbound, d_outbound, shape->x * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(inbound, d_inbound, shape->x * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(weights, d_weights, shape->x * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		print_result(outbound, inbound, weights, shape->x);
		//print_raw(outbound, inbound, weights);

		cout << runtime << " milliseconds." << endl;

		free_resources(inbound_vertices, outbound_vertices, shape, d_inbound_vertices, d_outbound_vertices, d_shape, d_red_array, outbound, inbound, weights, d_outbound, d_inbound, d_weights, d_current_node, d_last_node);

	}
}