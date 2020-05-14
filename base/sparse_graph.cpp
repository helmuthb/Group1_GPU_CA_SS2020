#include <string>
#include <stdexcept>
#include "sparse_graph.hpp"

static inline uint64_t xy2z(uint32_t x, uint32_t y) {
    return x | ((uint64_t)y << 32);
}

void SparseGraph::set(uint32_t x, uint32_t y, int32_t wNew) {
    bool hasValue = w.count(xy2z(x,y)) != 0;
    if (!hasValue && (wNew != WEIGHT_INFTY)) {
        m++;
    }
    else if (hasValue && (wNew == WEIGHT_INFTY)) {
        w.erase(xy2z(x, y));
        if (!directed) {
            w.erase(xy2z(y, x));
        }
        m--;
    }
    if (wNew != WEIGHT_INFTY) {
        w[xy2z(x,y)] = wNew;
        if (!directed) {
            w[xy2z(y,x)] = wNew;
        }
    }
}

int32_t SparseGraph::operator() (uint32_t x, uint32_t y) const {
    bool hasValue = w.count(xy2z(x,y)) != 0;
    return hasValue ? w.at(xy2z(x,y)) : WEIGHT_INFTY;
}

void SparseGraph::resize(uint32_t n0, uint32_t m0, bool d_flag) {
    Graph::resize(n0, m0, d_flag);
    w.clear();
    if (m0 > 0) {
        w.reserve(m0);
    }
}