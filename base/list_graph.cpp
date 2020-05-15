#include <string>
#include <stdexcept>
#include "list_graph.hpp"

void ListGraph::set(uint32_t x, uint32_t y, int32_t wNew) {
    bool hasValue = w[x].count(y) != 0;
    if (!hasValue && (wNew != WEIGHT_INFTY)) {
        m++;
    }
    else if (hasValue && (wNew == WEIGHT_INFTY)) {
        w[x].erase(y);
        if (!directed) {
            w[y].erase(x);
        }
        m--;
    }
    if (wNew != WEIGHT_INFTY) {
        w[x][y] = wNew;
        if (!directed) {
            w[y][x] = wNew;
        }
    }
}

void ListGraph::neighbors(uint32_t x, std::vector<EdgeTarget>& list) const {
    list.reserve(w[x].size());
    std::copy(w[x].begin(), w[x].end(), list.begin());
}

int32_t ListGraph::operator() (uint32_t x, uint32_t y) const {
    bool hasValue = w[x].count(y) != 0;
    return hasValue ? w[x].at(y) : WEIGHT_INFTY;
}

void ListGraph::resize(uint32_t n0, uint32_t m0, bool d_flag) {
    Graph::resize(n0, m0, d_flag);
    w.resize(n0);
}

ListGraph::ListGraph(const Graph& orig) : Graph(orig.num_vertices(), orig.is_directed()), w(orig.num_vertices()) {
    for (uint32_t i = 0; i < n; ++i) {
        std::vector<EdgeTarget> row;
        orig.neighbors(i, row);
        for (auto it = row.begin(); it < row.end(); ++it) {
            w[i][it->vertex_to] = it->weight;
        }
    }
}