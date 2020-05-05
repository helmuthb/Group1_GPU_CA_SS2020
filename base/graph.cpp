#include "graph.hpp"
#include <string>
#include <stdexcept>

const int Graph::WEIGHT_INFTY;

void Graph::set(int x, int y, int wNew) {
    int i = idx(x,y);
    int wOld = w[i];
    if (wOld == WEIGHT_INFTY && wNew != WEIGHT_INFTY) {
        m++;
    }
    else if (wOld != WEIGHT_INFTY && wNew == WEIGHT_INFTY) {
        m--;
    }
    w[i] = wNew;
    if (!directed) {
        // set also other vertice for undirected graph
        w[idx(y,x)] = wNew;
    }
}

std::ostream& operator<< (std::ostream& os, const Graph& g) {
    // header line
    os << "H " << g.num_vertices() << " " << g.num_edges() << " "
       << (g.is_directed()?"2":"1") << std::endl;
    // for each edge
    for (int i=0; i<g.num_vertices(); i++) {
        int j0 = (g.is_directed() ? 0 : i+1);
        for (int j=j0; j<g.num_vertices(); j++) {
            if (g(i,j) < Graph::WEIGHT_INFTY) {
                os << "E " << i << " " << j << " " << g(i,j) << std::endl;
            }
        }
    }
    return os;
}

std::istream& operator>> (std::istream& is, Graph& g) {
    // header line
    std::string word;
    is >> word;
    if (word != "H") {
        throw new std::runtime_error("Graph file does not start with 'H'");
    }
    int n, m, d_val;
    is >> n >> m >> d_val;
    // create temporary new graph
    Graph new_graph(n, (d_val==1)?false:true);
    for (int i=0; i<m; i++) {
        int x, y, w;
        is >> word >> x >> y >> w;
        if (word != "E") {
            throw new std::runtime_error("Graph edge does not start with 'E'");
        }
        new_graph.set(x, y, w);
    }
    g = new_graph;
    return is;
}
