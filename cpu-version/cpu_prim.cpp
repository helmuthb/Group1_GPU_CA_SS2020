#include "graph.hpp"
#include "cpu_prim.hpp"

int cpuNearestVertex(const Graph& g, const int d[], const bool v[]) {
    int min_d = Graph::WEIGHT_INFTY;
    int min_i = -1;
    for (int i=0; i<g.num_vertices(); i++) {
        if (!v[i] && d[i] < min_d) {
            min_d = d[i];
            min_i = i;
        }
    }
    return min_i;
}

Graph cpuPrimAlgorithm(const Graph& g) {
    int n = g.num_vertices();
    // store the vertices found so far, the distances & the predecessors
    bool *v = new bool[n];
    int *d = new int[n];
    int *p = new int[n];
    // initialize distances & vertices set
    for (int i=0; i<n; i++) {
        v[i] = false;
        d[i] = Graph::WEIGHT_INFTY;
        p[i] = -1;
    }
    // construct resulting graph
    Graph result(n, g.is_directed());
    // first vertex
    v[0] = true;
    d[0] = 0;
    for (int i=1; i<n; i++) {
        if (g(0,i) < Graph::WEIGHT_INFTY) {
            d[i] = g(0,i);
            p[i] = 0;
        }
    }
    // count vertices so far
    int cnt = 1;
    // outer while loop: till we have n vertices
    while (cnt < n) {
        // find vertex with minimum distance
        int v_next = cpuNearestVertex(g, d, v);
        if (v_next < 0) {
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
        for (int i=1; i<n; i++) {
            if (!v[i]) {
                if (d[i] > g(v_next, i)) {
                    d[i] = g(v_next, i);
                    p[i] = v_next;
                }
            }
        }
    }
    // free allocated memory
    delete[] v;
    delete[] d;
    delete[] p;
    // return resulting graph
    return result;
}