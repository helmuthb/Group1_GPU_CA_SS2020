/**
 * Graph Generator: generating random graph files
 */

#include <stdexcept>
#include <iostream>
#include <bits/stdc++.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

using namespace std;

// value to use for "infinity"
#define WEIGHT_INFTY INT_MAX

// sanity limit: number of nodes - 100K (corrsponds to 40GB RAM)
#define MAX_NODES 100000

class Graph {
private:
    unsigned int n_nodes;
    unsigned int n_edges;
    unsigned int *weight;
    // disable assignment operator
    const Graph& operator= (const Graph &) {}
    // get position for i, j
    unsigned int pos(int i, int j) const {
        int p = i + n_nodes*j;
        if (i >= n_nodes || j >= n_nodes || i < 0 || j < 0) {
            throw new std::out_of_range("Accessing invalid node");
        }
        return p;
    }
    // get random float from 0 to 1
    static float rnum() {
        return float(rand()) / RAND_MAX;
    }
public:
    // constructor
    Graph(int num_nodes, int min_weight, int max_weight, float density, bool directed) : n_nodes(num_nodes) {
        // check num_nodes
        if (num_nodes <= 0 || num_nodes > MAX_NODES) {
            throw new std::out_of_range("Maximum number of nodes exceeded");
        }
        // check weight range
        if (max_weight <= min_weight || min_weight < 0 || max_weight >= WEIGHT_INFTY) {
            throw new std::out_of_range("Weight range exceeded");
        }
        // check density
        if (density <= 0 || density >=1) {
            throw new std::out_of_range("Density range exceeded");
        }
        // allocate adjacency matrix
        weight = new unsigned int(n_nodes*n_nodes);
        // weight range
        int weight_range = max_weight - min_weight + 1;
        n_edges = 0;
        // loop through possible edges, i < j
        for (int i=0; i<n_nodes; i++) {
            (*this)(i,i) = WEIGHT_INFTY;
            for (int j=i+1; j<n_nodes; j++) {
                bool edge_exists = rnum() < density;
                unsigned int w = edge_exists ? min_weight + rnum()*weight_range : WEIGHT_INFTY;
                if (edge_exists) n_edges++;
                (*this)(i,j) = w;
                if (directed && edge_exists) {
                    // directed graph - other weights for other direction
                    w = min_weight + rnum()*weight_range;
                }
                (*this)(j,i) = w;
            }
        }
    }
    // copy constructor
    Graph(const Graph &org) : n_nodes(org.n_nodes), n_edges(org.n_edges) {
        // allocate memory
        weight = new unsigned int(n_nodes*n_nodes);
        // copy over weights
        memcpy(weight, org.weight, n_nodes*n_nodes*sizeof(int));
    }
    // destructor
    virtual ~Graph() {
        delete weight;
    }
    // get weight of edge
    unsigned int operator() (int i, int j) const {
        return weight[pos(i,j)];
    }
    // set weight of edge
    unsigned int & operator() (int i, int j) {
        return weight[pos(i,j)];
    }
    // output
    friend ostream & operator << (ostream & stream, const Graph & graph);
};

// write to file
ostream & operator << (ostream & stream, const Graph & graph) {
    // header line
    stream << "H " << graph.n_nodes << " " << graph.n_edges << " 1" << endl;
    // for each edge
    for (int i=0; i<graph.n_nodes; i++) {
        for (int j=i+1; j<graph.n_nodes; j++) {
            if (graph(i,j) < WEIGHT_INFTY) {
                stream << "E " << i << " " << j << " " << graph(i,j) << endl;
            }
        }
    }
    return stream;
}

extern "C" {
int main(int argc, const char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <nodes> <density> <min-weight> <max-weight>" << endl;
        return 1;
    }
    int num_nodes = atoi(argv[1]);
    float density = atof(argv[2]);
    int min_weight = atoi(argv[3]);
    int max_weight = atoi(argv[4]);
    Graph aGraph(num_nodes, min_weight, max_weight, density, false);
    // write to stdout
    cout << aGraph;
}
}