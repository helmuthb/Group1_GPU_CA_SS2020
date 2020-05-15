#ifndef __GRAPH_HPP
#define __GRAPH_HPP

#include <iostream>
#include <cstdint>
#include <utility>
#include <vector>

/**
 * This is a struct for describing the target of an edge.
 */
struct EdgeTarget {
    uint32_t vertex_to;
    int32_t weight;
    inline EdgeTarget() {};
    inline EdgeTarget(uint32_t v, int32_t w) : vertex_to(v), weight(w) {};
    inline EdgeTarget(std::pair<uint32_t, int32_t> p) : vertex_to(p.first), weight(p.second) {};
    inline const EdgeTarget& operator= (const std::pair<uint32_t, int32_t>& p) {
        vertex_to = p.first;
        weight = p.second;
        return (*this);
    }
};

/**
 * This is the base class to store, inspect & manipulate a graph.
 * It also allows reading and writing a graph.
 * 
 * There are two ways for accessing the edges of a graph:
 * 1) One can access via the adjacency matrix - providing the two vertices
 * 2) One can access them via a number from 0 to (max_edges). Here
 *    max_edges corresponds to the maximum number of edges possible.
 * Non-existing edges are stored as having indefinite weight, i.e.
 * Graph::WEIGHT_INFTY.
 * To get an edge the operator() is to be used.
 * To add / remove an edge the function set() is to be used.
 * Internally the object will keep track of the number of edges.
 * 
 * The class uses no memory allocation, therefore the standard
 * assignment operator and copy constructor are safe.
 */

class Graph {
protected:
    // number of vertices & edges
    uint32_t n, m;
    // flag whether the graph is directed or undirected
    bool directed;
    // more complex way to get x, y from the position in the
    // list of all possible edges
    void getEdge (uint64_t p, uint32_t& x, uint32_t& y) const;
public:
    // WEIGHT_INFTY denotes a non-existing edge
    static constexpr int32_t WEIGHT_INFTY = INT32_MAX;
    // operator(): to get the weight of an edge, in two variants
    virtual int operator() (uint32_t x, uint32_t y) const = 0;
    virtual int operator() (uint64_t p) const;
    // function set: add / remove / change weight of an edge
    virtual void set(uint32_t x, uint32_t y, int32_t wNew) = 0;
    virtual void set(uint64_t p, int32_t wNew);
    // return incident edges for a vertex(unsorted)
    virtual void neighbors(uint32_t x, std::vector<EdgeTarget>& list) const;
    // return the number of vertices
    uint32_t num_vertices() const { return n; }
    // return the number of edges (dynamically counted)
    uint32_t num_edges() const { return m; }
    // flag whether the graph is directed or undirected
    bool is_directed() const { return directed; }
    // resize the graph - specify expected number of edges in second param
    virtual void resize(uint32_t n0, uint32_t m0=0, bool d_flag=false);
    // constructor for a graph: number of vertices, flag for directed
    Graph(uint32_t n0=0, bool d_flag=false) : n(n0), m(0), directed(d_flag) {};
};

// write to file
std::ostream& operator<< (std::ostream& os, const Graph& g);

// read from file
std::istream& operator>> (std::istream& is, Graph& g);

#endif