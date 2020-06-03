#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "graph.hpp"
#include "matrix_graph.hpp"
#include "sparse_graph.hpp"
#include "list_graph.hpp"
#include "generator.hpp"
#include <sstream>
#include <string>
#include <iostream>
#include <cstdint>

TEST_CASE_TEMPLATE("creating a small undirected graph", T_GRAPH, MatrixGraph, SparseGraph, ListGraph) {
    T_GRAPH g;
    generator(g, 10, 2, 5, 0.5, false);
    CHECK(g.num_vertices() == 10);
    CHECK(g.is_directed() == false);
    WARN_MESSAGE(g.num_edges() > 15, "we expect about 22+5 edges");
    WARN_MESSAGE(g.num_edges() < 35, "we expect about 22+5 edges");
    int32_t max_w = 0, min_w = 100;
    for (uint32_t i=0; i<9; i++) {
        for (uint32_t j=i+1; j<10; j++) {
            if (g(i,j) != Graph::WEIGHT_INFTY) {
                if (g(i,j) > max_w) {
                    max_w = g(i,j);
                }
                if (g(i,j) < min_w) {
                    min_w = g(i,j);
                }
            }
        }
    }
    CHECK(max_w <= 5);
    CHECK(min_w >= 2);
    WARN_MESSAGE(max_w == 5, "we expect the maximum to be taken");
    WARN_MESSAGE(min_w == 2, "we expect the minimum to be taken");
}

TEST_CASE_TEMPLATE("creating a small directed graph", T_GRAPH, MatrixGraph, SparseGraph, ListGraph) {
    T_GRAPH g;
    generator(g, 10, 2, 5, 0.5, true);
    CHECK(g.num_vertices() == 10);
    CHECK(g.is_directed() == true);
    WARN_MESSAGE(g.num_edges() > 45, "we expect about 45+5 edges");
    WARN_MESSAGE(g.num_edges() < 55, "we expect about 45+5 edges");
    int32_t max_w = 0, min_w = 100;
    for (uint32_t i=0; i<10; i++) {
        for (uint32_t j=0; j<10; j++) {
            if (g(i,j) != Graph::WEIGHT_INFTY) {
                if (g(i,j) > max_w) {
                    max_w = g(i,j);
                }
                if (g(i,j) < min_w) {
                    min_w = g(i,j);
                }
            }
        }
    }
    CHECK(max_w <= 5);
    CHECK(min_w >= 2);
    WARN_MESSAGE(max_w == 5, "we expect the maximum to be taken");
    WARN_MESSAGE(min_w == 2, "we expect the minimum to be taken");
}

TEST_CASE_TEMPLATE("creating a large graph", T_GRAPH, MatrixGraph, SparseGraph, ListGraph) {
    T_GRAPH g;
    generator(g, 10000, 2, 5, 0.5, false);
    CHECK(g.num_vertices() == 10000);
    CHECK(g.is_directed() == false);
    WARN_MESSAGE(g.num_edges() > 24000000, "we expect about 25 mio edges");
    WARN_MESSAGE(g.num_edges() < 26000000, "we expect about 25 mio edges");
    int32_t max_w = 0, min_w = 100;
    for (uint32_t i=0; i<9999; i++) {
        for (uint32_t j=i+1; j<10000; j++) {
            if (g(i,j) != Graph::WEIGHT_INFTY) {
                if (g(i,j) > max_w) {
                    max_w = g(i,j);
                }
                if (g(i,j) < min_w) {
                    min_w = g(i,j);
                }
            }
        }
    }
    CHECK(max_w <= 5);
    CHECK(min_w >= 2);
    WARN_MESSAGE(max_w == 5, "we expect the maximum to be taken");
    WARN_MESSAGE(min_w == 2, "we expect the minimum to be taken");
}
