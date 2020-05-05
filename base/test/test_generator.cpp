#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "graph.hpp"
#include "generator.hpp"
#include <sstream>
#include <string>
#include <iostream>

TEST_CASE("creating a small undirected graph") {
    Graph g = generator(10, 2, 5, 0.5, false);
    CHECK(g.num_vertices() == 10);
    CHECK(g.is_directed() == false);
    WARN_MESSAGE(g.num_edges() > 10, "we expect about 22 edges");
    WARN_MESSAGE(g.num_edges() < 30, "we expect about 22 edges");
    int max_w = 0, min_w = 100;
    for (int i=0; i<9; i++) {
        for (int j=i+1; j<10; j++) {
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

TEST_CASE("creating a small directed graph") {
    Graph g = generator(10, 2, 5, 0.5, true);
    CHECK(g.num_vertices() == 10);
    CHECK(g.is_directed() == true);
    WARN_MESSAGE(g.num_edges() > 40, "we expect about 45 edges");
    WARN_MESSAGE(g.num_edges() < 50, "we expect about 45 edges");
    int max_w = 0, min_w = 100;
    for (int i=0; i<10; i++) {
        for (int j=0; j<10; j++) {
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

TEST_CASE("creating a large graph") {
    Graph g = generator(10000, 2, 5, 0.5, false);
    CHECK(g.num_vertices() == 10000);
    CHECK(g.is_directed() == false);
    WARN_MESSAGE(g.num_edges() > 24000000, "we expect about 25 mio edges");
    WARN_MESSAGE(g.num_edges() < 26000000, "we expect about 25 mio edges");
    int max_w = 0, min_w = 100;
    for (int i=0; i<9999; i++) {
        for (int j=i+1; j<10000; j++) {
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