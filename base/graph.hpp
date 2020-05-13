#ifndef __GRAPH_HPP
#define __GRAPH_HPP

#include <vector>
#include <iostream>
#include <limits.h>

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
private:
    // number of vertices & edges
    unsigned int n, m;
    // flag whether the graph is directed or undirected
    bool directed;
    // weights for each position
    std::vector<int> w;
    // get index for x,y in the vector
    int idx(int x, int y) const { return x+n*y; }
    // more complex way to get x, y from the position in the
    // list of all possible edges
    void getEdge (unsigned long long p, int& x, int& y) const;
public:
    // WEIGHT_INFTY denotes a non-existing edge
    static constexpr int WEIGHT_INFTY = INT_MAX;
    // operator(): to get the weight of an edge, in two variants
    int operator() (int x, int y) const { return w[idx(x,y)]; }
    int operator() (unsigned long long p) const;
    // function set: add / remove / change weight of an edge
    void set(int x, int y, int wNew);
    void set(unsigned long long p, int wNew);
    // return the number of vertices (cannot be changed)
    int num_vertices() const { return n; }
    // return the number of edges (dynamically counted)
    int num_edges() const { return m; }
    // flag whether the graph is directed or undirected
    bool is_directed() const { return directed; }
    // constructor for a graph: number of vertices, flag for directed
    Graph(int n0=0, bool d_flag=false) : n(n0), m(0), directed(d_flag), w(std::vector<int>(n0*n0, WEIGHT_INFTY)) {};
};

// write to file
std::ostream& operator<< (std::ostream& os, const Graph& g);

// read from file
std::istream& operator>> (std::istream& is, Graph& g);

#endif