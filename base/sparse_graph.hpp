#ifndef __SPARSE_GRAPH_HPP
#define __SPARSE_GRAPH_HPP

#include "graph.hpp"
#include <unordered_map>
#include <cstdint>


/**
 * This is a sparse-matrix based implementation of the
 * base class Graph.
 * 
 * It is memory efficent, especially for sparse graphs.
 * The runtime performance is very poor - 10x slower than the
 * matrix based class.
 * The class uses no memory allocation, therefore the standard
 * assignment operator and copy constructor are safe.
 */
class SparseGraph : public Graph {
protected:
    // weights for each position
    std::unordered_map<uint64_t, int32_t> w;
public:
    // operator(): to get the weight of an edge
    using Graph::operator();
    virtual int32_t operator() (uint32_t x, uint32_t y) const override;
    // function set: add / remove / change weight of an edge
    using Graph::set;
    virtual void set(uint32_t x, uint32_t y, int32_t wNew) override;
    virtual void resize(uint32_t n0, uint32_t m0=0, bool d_flag=false) override;
    SparseGraph(uint32_t n0=0, bool d_flag=false) : Graph(n0, d_flag), w({}) {};
};

#endif