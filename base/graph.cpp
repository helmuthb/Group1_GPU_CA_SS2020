#include "graph.hpp"

const int32_t Graph::WEIGHT_INFTY;

void Graph::set(uint64_t p, int32_t wNew) {
    uint32_t x, y;
    getEdge(p, x, y);
    set(x, y, wNew);
}

void Graph::getEdge(uint64_t p, uint32_t& x, uint32_t& y) const {
    if (directed) {
        if (p >= n*(n-1)) {
            throw new std::out_of_range("Maximum number of edges exceeded");
        }
        y = p / (n-1);
        x = p % (n-1);
        if (x >= y) {
            x++;
        }
    }
    else {
        if (p >= n*(n-1)/2) {
            throw new std::out_of_range("Maximum number of edges exceeded");
        }
        y = p / (n-1);
        x = p % (n-1);
        if (x >= y) {
            x = n - x - 2;
            y = n - y - 1;
        }
    }
}

int32_t Graph::operator() (uint64_t p) const {
    uint32_t x, y;
    const Graph& t = *this;
    getEdge(p, x, y);
    return t(x, y);
}

void Graph::neighbors(uint32_t x, std::vector<EdgeTarget>& list) const {
    const Graph& t = *this;
    for (uint32_t i=0; i<n; i++) {
        int32_t weight = t(x,i);
        if (weight != WEIGHT_INFTY) {
            list.push_back(EdgeTarget(i, weight));
        }
    }
}

void Graph::resize(uint32_t n0, uint32_t m0, bool d_flag) {
    n = n0;
    m = 0;
    directed = d_flag;
}

std::ostream& operator<< (std::ostream& os, const Graph& g) {
    // header line
    os << "H " << g.num_vertices() << " " << g.num_edges() << " "
       << (g.is_directed()?"2":"1") << std::endl;
    // for each edge
    for (uint32_t i=0; i<g.num_vertices(); i++) {
        uint32_t j0 = (g.is_directed() ? 0 : i+1);
        for (uint32_t j=j0; j<g.num_vertices(); j++) {
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
    uint32_t n, m, d_val;
    is >> n >> m >> d_val;
    // resize graph
    g.resize(n, m, d_val==2);
    for (uint32_t i=0; i<m; i++) {
        uint32_t x, y;
        int32_t w;
        is >> word >> x >> y >> w;
        if (word != "E") {
            throw new std::runtime_error("Graph edge does not start with 'E'");
        }
        g.set(x, y, w);
    }
    return is;
}
