#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cstdint>
#include "graph.hpp"
#include "thrust_prim.hpp"

void thrustPrepare(const Graph& g, thrust::host_vector<uint32_t>* num_edges, thrust::host_vector<uint32_t>* idx_edges,
                   thrust::host_vector<uint32_t>* target, thrust::host_vector<int32_t>* weight) {
    uint32_t n = g.num_vertices();
    uint32_t m = g.num_edges();
    (*num_edges) = thrust::host_vector<uint32_t>(n);
    (*idx_edges) = thrust::host_vector<uint32_t>(n);
    (*target) = thrust::host_vector<uint32_t>(g.is_directed() ? m : 2*m);
    (*weight) = thrust::host_vector<int32_t>(g.is_directed() ? m : 2*m);
    uint32_t pos = 0;
    for (uint32_t v = 0; v<n; ++v) {
        std::vector<EdgeTarget> neighbors;
        g.neighbors(v, neighbors);
        (*num_edges)[v] = neighbors.size();
        (*idx_edges)[v] = pos;
        for (auto nb = neighbors.begin(); nb<neighbors.end(); ++nb) {
            (*target)[pos] = nb->vertex_to;
            (*weight)[pos++] = nb->weight; 
        }
    }
}

uint32_t thrustNearestVertex(thrust::device_vector<int32_t>& distances) {
    auto minPos = thrust::min_element(distances.begin(), distances.end());
    return minPos - distances.begin();
}

static int32_t constexpr DIST_IN_MST = Graph::WEIGHT_INFTY;
static int32_t constexpr DIST_DEFAULT = Graph::WEIGHT_INFTY-1;
static uint32_t constexpr VERTEX_NONE = UINT32_MAX;

void thrustPrimAlgorithm(thrust::host_vector<uint32_t>* num_neighbors, thrust::host_vector<uint32_t>* idx_edges,
                        thrust::host_vector<uint32_t>* target, thrust::host_vector<int32_t>* weight,
                        thrust::host_vector<uint32_t>* predecessor) {
    // number of vertices = size of the "num_edges" array
    uint32_t n = num_neighbors->size();
    uint32_t m = target->size();
                        
    // copy the graph info to the device
    thrust::device_vector<uint32_t> d_num_edges = *num_neighbors;
    thrust::device_vector<uint32_t> d_idx_edges = *idx_edges;
    thrust::device_vector<uint32_t> d_target = *target;
    thrust::device_vector<int32_t> d_weight = *weight;
    
    // distances vector
    thrust::device_vector<int32_t> distances(n, DIST_DEFAULT);
    // initialize predecessor vector
    (*predecessor) = thrust::device_vector<uint32_t>(n, VERTEX_NONE);
    for (uint32_t cnt = 0; cnt < n; ++cnt) {
        uint32_t nextVertex = (cnt == 0) ? 0 : thrustNearestVertex(distances);
        // mark vertex as already in MST
        distances[nextVertex] = DIST_IN_MST;
        // update distances:
        // number of neighbors of nextVertex is in num_neighbors[nextVertex]
        // first neighbor's index is in idx_edges[nextVertex]
        uint32_t numNeighbors = (*num_neighbors)[nextVertex];
        uint32_t idxFirstNeighbor = (*idx_edges)[nextVertex];
        for (uint32_t i=0; i<numNeighbors; ++i) {
            uint32_t otherVertex = (*target)[idxFirstNeighbor + i];
            int32_t theWeight = (*weight)[idxFirstNeighbor + i];
            if (distances[otherVertex] != DIST_IN_MST) {
                if (distances[otherVertex] > theWeight) {
                    distances[otherVertex] = theWeight;
                    (*predecessor)[otherVertex] = nextVertex;
                }
            }
        }
    }
}