#ifndef __MATRIX_GRAPH_HPP
#define __MATRIX_GRAPH_HPP

#include "graph.hpp"
#include <vector>
#include <cstdint>

/**
 * This is an adjacency-matrix based implementation of the
 * base class Graph.
 * 
 * It is runtime-performant, but not efficiently using memory,
 * especially for sparse graphs.
 * The class uses no memory allocation, therefore the standard
 * assignment operator and copy constructor are safe.
 */
class MatrixGraph : public Graph {
protected:
    // weights for each position
    std::vector<int32_t> w;
    // get index for x,y in the vector
    uint64_t idx(uint32_t x, uint32_t y) const { return x+n*y; }
public:
    // operator(): to get the weight of an edge
    using Graph::operator();
    virtual int32_t operator() (uint32_t x, uint32_t y) const override;
    // function set: add / remove / change weight of an edge
    using Graph::set;
    virtual void set(uint32_t x, uint32_t y, int32_t wNew) override;
    virtual void resize(uint32_t n0, uint32_t m0=0, bool d_flag=false) override;
    MatrixGraph(uint32_t n0=0, bool d_flag=false) : Graph(n0, d_flag), w(std::vector<int32_t>(n0*n0, WEIGHT_INFTY)) {};
    MatrixGraph(const Graph&);
};

#endif