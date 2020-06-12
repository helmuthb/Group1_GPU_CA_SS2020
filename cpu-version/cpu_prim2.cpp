#include "graph.hpp"
#include "cpu_prim2.hpp"
#include <cstdint>
#include <utility>
#include <algorithm>
#include <vector>

void cpuPrim2Algorithm(const Graph& g, Graph& mst) {
    // allocate 3 arrays OUT, IN, W, each of length |V|-1
    auto nV = g.num_vertices();
    auto OUT = std::vector<u_int32_t>(nV-1);
    auto IN = std::vector<u_int32_t>(nV-1);
    auto W = std::vector<int32_t>(nV-1);
    // helper table - to find inverse from IN
    auto IN_reverse = std::vector<u_int32_t>(nV);
    IN_reverse[0] = 0;
    // pick a random initial vertex C (e.g. 0)
    uint32_t C = 0;
    for (u_int32_t i=0; i<nV-1; i++) {
        OUT[i] = C;
        IN[i] = i+1; // connect all other vertices to C
        IN_reverse[i+1] = i;
        W[i] = g(C, i+1);
    }
    // The arrays OUT, IN, W now contain every hypothetical edge
    // from the initial vertex C (stored in OUT) to all other vertices (in IN).
    // If that edge actually exists, then W[] is < INFTY.

    // Steps
    // The three arrays are “split” into two parts:
    // (1) edges done, and (2) edges remaining.
    // The split is determined by some offset O, initially this is 0 (all edges are remaining).
    // The offset is incremented by 1 every iteration.
    uint32_t O = 0;

    // For the sake of explanation, we call the
    // remaining sets OUT’, IN’ and W’
    for (uint32_t i=0; i<nV-1; i++) {
        // FIND Step
        // Find the minimum weighted edge in W’
        int32_t minEdgeWeight = Graph::WEIGHT_INFTY;
        uint32_t minEdgeIndex = O;
        for (uint32_t j=O; j<nV-1; j++) {
            if (W[j] < minEdgeWeight) {
                minEdgeWeight = W[j];
                minEdgeIndex = j;
            }
        }
        // MOVE Step
        // If that edge is not the first in W’, then swap those two edges
        if (minEdgeIndex != O) {
            W[minEdgeIndex] = W[O];
            W[O] = minEdgeWeight;
            auto swapVertex = OUT[minEdgeIndex];
            OUT[minEdgeIndex] = OUT[O];
            OUT[O] = swapVertex;
            swapVertex = IN[minEdgeIndex];
            IN[minEdgeIndex] = IN[O];
            IN[O] = swapVertex;
            IN_reverse[IN[O]] = O;
            IN_reverse[IN[minEdgeIndex]] = minEdgeIndex;
            minEdgeIndex = O;
        }
        // Update C = vertex connected by the edge
        C = IN[O];
        // UPDATE Step
        // Remove the new edge from W’ (= increase the starting offset of W’ by 1)
        O++;
        // For each vertex v remaining in IN’:
        // If there exists (in the global edge table!) an edge C->v and its weight is lesser
        // than the current weight in W’ to reach v, then set the OUT vertex to C, and update the weight.
        std::vector<EdgeTarget> fromC;
        g.neighbors(C, fromC);
        for (auto other : fromC) {
            auto j = IN_reverse[other.vertex_to];
            if (j >= O && W[j] > other.weight) {
                W[j] = other.weight;
                OUT[j] = C;
            }
        }
    }
    // Output the MST into the argument
    mst.resize(nV, nV-1, false);
    for (uint32_t i=0; i<nV-1; i++) {
        // mst.set(OUT[i], IN[i], W[i]);
        mst.set(OUT[i], IN[i], g(OUT[i], IN[i]));
    }
}
