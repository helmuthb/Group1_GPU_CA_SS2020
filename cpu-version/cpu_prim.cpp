#include "graph.hpp"
#include "cpu_prim.hpp"
#include <cstdint>
#include <utility>
#include <algorithm>

class vertex_distance {
public:
    int32_t distance;
    uint32_t predecessor;
    vertex_distance() : distance(Graph::WEIGHT_INFTY), predecessor(UINT32_MAX) {};
};

class distance_heap {
protected:
    std::vector<vertex_distance> vd;
    std::vector<uint32_t> heap;
    std::vector<uint32_t> inverse;
    bool compare_i (uint32_t i1, uint32_t i2) const {
        return vd[heap[i1]].distance > vd[heap[i2]].distance;
    };
    // parent of i
    uint32_t parent(uint32_t i) { return i==0?0:(i-1)/2; }
    // child 1
    uint32_t child1(uint32_t i) { return 2*i+1; }
    // child 2
    uint32_t child2(uint32_t i) { return 2*i+2; }
    // swap 2
    void swap(uint32_t kid, uint32_t dad) {
        uint32_t oldKid = heap[kid];
        heap[kid] = heap[dad];
        heap[dad] = oldKid;
        // swap inverse as well
        uint32_t inv_oldKid = inverse[heap[kid]];
        inverse[heap[kid]] = inverse[heap[dad]];
        inverse[heap[dad]] = inv_oldKid;
    }
    // bubble up
    void bubbleUp(uint32_t kid) {
        uint32_t dad = parent(kid);
        while (compare_i(dad, kid)) {
            swap(kid, dad);
            kid = dad;
            dad = parent(kid);
        }
    }
    // bubble down
    void bubbleDown(uint32_t dad) {
        uint32_t size = heap.size();
        while (true) {
            uint32_t left = child1(dad);
            uint32_t right = child2(dad);
            uint32_t nearest = dad;
            if (left < size && compare_i(nearest, left)) {
                nearest = left;
            }
            if (right < size && compare_i(nearest, right)) {
                nearest = right;
            }
            if (nearest == dad) {
                return;
            }
            swap(nearest, dad);
            dad = nearest;
        }
    }
public:
    distance_heap(uint32_t n) : vd(n), heap(n), inverse(n) {
        for (uint32_t i=0; i<n; i++) {
            heap[i] = i;
            inverse[i] = i;
        }
    };
    uint32_t pop_nearest() {
        // the first is the smallest
        uint32_t vmin = heap[0];
        uint32_t last = heap.size()-1;
        // swap first with last
        swap(0, last);
        // remove last element (previously first)
        heap.pop_back();
        // bubble down from the top
        bubbleDown(0);
        return vmin;
    };
    int32_t distance(uint32_t vertex) const { return vd[vertex].distance; }
    uint32_t predecessor(uint32_t vertex) const { return vd[vertex].predecessor; }
    void reduce_if(uint32_t vertex, int32_t distance, uint32_t predecessor) {
        uint32_t inv = inverse[vertex];
        if (inv < heap.size()) {
            if (vd[vertex].distance > distance) {
                vd[vertex].distance = distance;
                vd[vertex].predecessor = predecessor;
                // reorganize heap
                bubbleUp(inv);
            }
        }
    };
};

void cpuPrimAlgorithm(const Graph& g, Graph& mst) {
    uint32_t n = g.num_vertices();
    distance_heap dheap(n);
    // resize resulting mst
    mst.resize(n, g.is_directed());
    for (uint32_t i=0; i<n; i++) {
        // get next vertex
        uint32_t vnext = dheap.pop_nearest();
        if (dheap.predecessor(vnext) != UINT32_MAX) {
            // add to MST
            mst.set(dheap.predecessor(vnext), vnext, dheap.distance(vnext));
        }
        // update all distances
        std::vector<EdgeTarget> neighbors;
        g.neighbors(vnext, neighbors);
        // std::vector<std::pair<uint32_t, int32_t>> neighbors = g.neighbors(vnext);
        for (auto nb = neighbors.begin(); nb<neighbors.end(); nb++) {
            dheap.reduce_if(nb->vertex_to, nb->weight, vnext);
        }
    }
}