#include "graph.hpp"
#include "cpu_prim.hpp"
#include <cstdint>

int cpuNearestVertex(const Graph& g, const int32_t d[], const bool v[]) {
    int32_t min_d = Graph::WEIGHT_INFTY;
    uint32_t min_i = UINT32_MAX;
    for (uint32_t i=0; i<g.num_vertices(); i++) {
        if (!v[i] && d[i] < min_d) {
            min_d = d[i];
            min_i = i;
        }
    }
    return min_i;
}

void cpuPrimAlgorithm(const Graph& g, Graph& result) {
    uint32_t n = g.num_vertices();
    // store the vertices found so far, the distances & the predecessors
    bool *v = new bool[n];
    int32_t *d = new int32_t[n];
    uint32_t *p = new uint32_t[n];
    // initialize distances & vertices set
    for (uint32_t i=0; i<n; i++) {
        v[i] = false;
        d[i] = Graph::WEIGHT_INFTY;
        p[i] = UINT32_MAX;
    }
    // resize resulting graph
    result.resize(n, n-1, g.is_directed());
    // first vertex
    v[0] = true;
    d[0] = 0;
    for (uint32_t i=1; i<n; i++) {
        if (g(0,i) < Graph::WEIGHT_INFTY) {
            d[i] = g(0,i);
            p[i] = 0;
        }
    }
    // count vertices so far
    uint32_t cnt = 1;
    // outer while loop: till we have n vertices
    while (cnt < n) {
        // find vertex with minimum distance
        uint32_t v_next = cpuNearestVertex(g, d, v);
        if (v_next == UINT32_MAX) {
            throw new std::runtime_error("Did not find any vertex");
        }
        // add to set
        v[v_next] = true;
        cnt++;
        // add edge to graph
        // if we don't find an edge then the original graph was not connected
        if (p[v_next] >= 0) {
            result.set(p[v_next], v_next, g(p[v_next], v_next));
        }
        // update all distances
        std::vector<EdgeTarget> neighbors;
        g.neighbors(v_next, neighbors);
        uint32_t nb_count = neighbors.size();
        for (uint32_t j=0; j<nb_count; j++) {
            EdgeTarget nb = neighbors[j];
            if (!v[nb.vertex_to]) {
                if (d[nb.vertex_to] > nb.weight) {
                    d[nb.vertex_to] = nb.weight;
                    p[nb.vertex_to] = v_next;
                }
            }
        }
    }
    // free allocated memory
    delete[] v;
    delete[] d;
    delete[] p;
}