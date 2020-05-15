#include <string>
#include <stdexcept>
#include "matrix_graph.hpp"

void MatrixGraph::set(uint32_t x, uint32_t y, int32_t wNew) {
    uint64_t i = idx(x,y);
    int32_t wOld = w[i];
    if (wOld == WEIGHT_INFTY && wNew != WEIGHT_INFTY) {
        m++;
    }
    else if (wOld != WEIGHT_INFTY && wNew == WEIGHT_INFTY) {
        m--;
    }
    w[i] = wNew;
    if (!directed) {
        // set also other edge for undirected graph
        w[idx(y,x)] = wNew;
    }
}

int32_t MatrixGraph::operator() (uint32_t x, uint32_t y) const {
    return w[idx(x,y)];
}

void MatrixGraph::resize(uint32_t n0, uint32_t m0, bool d_flag) {
    Graph::resize(n0, m0, d_flag);
    w = std::vector<int32_t>(n0*n0, WEIGHT_INFTY);
}

MatrixGraph::MatrixGraph(const Graph& orig) : Graph(orig.num_vertices(), orig.is_directed()) {
    w = std::vector<int32_t>(n*n, WEIGHT_INFTY);
    std::vector<Edge> e(m);
    this->edges(e);
    for (auto it = e.begin(); it < e.end(); ++it) {
        w[idx(it->vertex_from, it->vertex_to)] = it->weight;
    }
}