#include "graph.hpp"


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

