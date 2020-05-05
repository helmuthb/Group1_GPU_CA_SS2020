#ifndef __GRAPH_HPP
#define __GRAPH_HPP

#include <vector>
#include <iostream>
#include <limits.h>

class Graph {
private:
    unsigned int n, m;
    bool directed;
    std::vector<int> w;
    int idx(int x, int y) const { return x+n*y; }
public:
    static constexpr int WEIGHT_INFTY = INT_MAX;
    int operator() (int x, int y) const { return w[idx(x,y)]; }
    void set(int x, int y, int w0);
    int num_vertices() const { return n; }
    int num_edges() const { return m; }
    bool is_directed() const { return directed; }
    Graph(int n0=0, bool d_flag=false) : n(n0), m(0), directed(d_flag), w(std::vector<int>(n0*n0, WEIGHT_INFTY)) {};
};

// write to file
std::ostream& operator<< (std::ostream& os, const Graph& g);

// read from file
std::istream& operator>> (std::istream& is, Graph& g);

#endif