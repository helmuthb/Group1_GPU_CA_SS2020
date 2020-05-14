#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "cpu_prim.hpp"
#include "matrix_graph.hpp"
#include "sparse_graph.hpp"
#include <vector>

TEST_CASE_TEMPLATE("prim for tiny graph", T_GRAPH, MatrixGraph, SparseGraph) {
    T_GRAPH g(6, false);
    g.set(0,1,1);
    g.set(0,2,3);
    g.set(0,5,2);
    g.set(1,2,5);
    g.set(1,3,1);
    g.set(2,3,2);
    g.set(2,4,1);
    g.set(3,4,4);
    g.set(4,5,5);
    T_GRAPH mst;
    cpuPrimAlgorithm(g, mst);
    CHECK(mst.num_vertices() == 6);
    CHECK(mst.num_edges() == 5);
    CHECK(mst(0,1) == 1);
    CHECK(mst(0,5) == 2);
    CHECK(mst(0,2) == Graph::WEIGHT_INFTY);
    CHECK(mst(1,3) == 1);
    CHECK(mst(1,2) == Graph::WEIGHT_INFTY);
    CHECK(mst(2,4) == 1);
    CHECK(mst(3,2) == 2);
}

TEST_CASE_TEMPLATE("prim for small graph", T_GRAPH, MatrixGraph, SparseGraph) {
    T_GRAPH g(10, false);
    g.set(0, 1, 3);
    g.set(0, 3, 6);
    g.set(0, 5, 9);
    g.set(1, 3, 4);
    g.set(1, 2, 2);
    g.set(1, 4, 9);
    g.set(1, 5, 9);
    g.set(2, 3, 2);
    g.set(2, 4, 8);
    g.set(2, 6, 9);
    g.set(3, 6, 9);
    g.set(4, 6, 7);
    g.set(4, 8, 9);
    g.set(4, 9, 10);
    g.set(4, 5, 8);
    g.set(5, 9, 18);
    g.set(6, 7, 4);
    g.set(6, 8, 5);
    g.set(7, 8, 1);
    g.set(7, 9, 4);
    g.set(8, 9, 3);
    T_GRAPH mst;
    cpuPrimAlgorithm(g, mst);
    CHECK(mst.num_vertices() == 10);
    CHECK(mst.num_edges() == 9);
    CHECK(mst(0, 1) == 3);
    CHECK(mst(1, 2) == 2);
    CHECK(mst(2, 3) == 2);
    CHECK(mst(2, 4) == 8);
    CHECK(mst(4, 6) == 7);
    CHECK(mst(4, 5) == 8);
    CHECK(mst(6, 7) == 4);
    CHECK(mst(7, 8) == 1);
    CHECK(mst(8, 9) == 3);
}