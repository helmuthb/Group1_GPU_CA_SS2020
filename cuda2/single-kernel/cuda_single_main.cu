#include <iostream>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "matrix_graph.hpp"
#include "generator.hpp"
#include "graph.hpp"
#include "kernel.cu"
#include "array_structure_helper.cu"

#include <chrono>

#define NUM_RUNS 1

using namespace std;

void print_result(uint32_t * outbound, uint32_t *inbound, uint32_t *weights, uint32_t V) {
	cout << "H " << V << " " << V - 1 << " " << 1 << endl;
	for (int i = 0; i < V; i++) {
		if (weights[i] != UINT32_MAX) {
			cout << "E " << outbound[i] << " " << inbound[i] << " " << weights[i] << endl;
		}
	}
}

int main()
{
	MatrixGraph aGraph;

	chrono::steady_clock::time_point begin, end;
	double runtime;

	for (int i = 0; i < NUM_RUNS; i++) {

		uint2 * inbound_vertices, *outbound_vertices, *shape = NULL;
		uint2 * d_inbound_vertices = NULL, *d_outbound_vertices, *d_shape;

		uint32_t *d_outbound, *d_inbound, *d_weights;

		generator(aGraph, 100, 0, 100, 0.7, false);
		//cin >> aGraph;

		// write to stdout
		cout << aGraph << endl;

		cudaSetup(aGraph, inbound_vertices, outbound_vertices, shape);

		const uint32_t V = shape->x;
		const uint32_t E = shape->y;

		uint32_t *outbound = new uint32_t[V];
		uint32_t *inbound = new uint32_t[V];
		uint32_t *weights = new uint32_t[V];

		// start node
		uint32_t current_node = 0, *d_current_node = NULL;

		std::fill_n(weights, V, UINT32_MAX);

		uint32_t *d_num_edges, *d_idx_edges, *d_target, *d_weight;

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

		begin = chrono::steady_clock::now();
		mst << <1, 1024 >> > (d_inbound_vertices, d_outbound_vertices, d_shape, d_inbound, d_outbound, d_weights, d_current_node);
		end = chrono::steady_clock::now();
		runtime = (chrono::duration_cast<chrono::duration<double>>(end - begin)).count() * 1000;

		cout << runtime << endl;

		cudaMemcpy(outbound, d_outbound, V * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(inbound, d_inbound, V * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(weights, d_weights, V * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		print_result(outbound, inbound, weights, V);

		cudaFree(d_inbound_vertices);
		cudaFree(d_outbound_vertices);
		cudaFree(d_shape);

		cudaFree(d_inbound);
		cudaFree(d_outbound);
		cudaFree(d_weights);
		cudaFree(d_current_node);

		delete[] inbound_vertices;
		delete[] outbound_vertices;
		delete[] shape;

		delete[] inbound;
		delete[] outbound;
		delete[] weights;
	}
}
