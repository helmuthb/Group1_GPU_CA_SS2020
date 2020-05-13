#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "graph.hpp"
#include <sstream>
#include <string>
#include <iostream>

TEST_CASE("creating a graph") {
    Graph g(10, false);
    CHECK(g.num_vertices() == 10);
    CHECK(g.num_edges() == 0);
    CHECK(g.is_directed() == false);
}

TEST_CASE("adding an edge in an undirected graph") {
    Graph g(10, false);
    g.set(1, 2, 5);
    CHECK(g(0,1) == Graph::WEIGHT_INFTY);
    CHECK(g(1,2) == 5);
    CHECK(g(2,1) == 5);
    CHECK(g.num_vertices() == 10);
    CHECK(g.num_edges() == 1);
}

TEST_CASE("adding all possible edges of an undirected graph") {
    Graph g(10, false);
    for (int i=0; i<10*9/2; i++) {
        g.set(i, 1);
    }
    CHECK(g(0,1) == 1);
    CHECK(g.num_edges() == 45);
}


TEST_CASE("adding all possible edges of a directed graph") {
    Graph g(10, true);
    for (int i=0; i<10*9; i++) {
        g.set(i, 1);
    }
    CHECK(g(0,1) == 1);
    CHECK(g.num_edges() == 90);
}

TEST_CASE("adding an edge in a directed graph") {
    Graph g(10, true);
    g.set(1, 2, 5);
    CHECK(g(0,1) == Graph::WEIGHT_INFTY);
    CHECK(g(1,2) == 5);
    CHECK(g(2,1) == Graph::WEIGHT_INFTY);
    CHECK(g.num_vertices() == 10);
    CHECK(g.num_edges() == 1);
}

TEST_CASE("reading an undirected graph") {
    Graph g;
    std::string str = "H 3 2 1\nE 0 1 3\nE 1 2 5\n";
    std::stringstream(str) >> g;
    CHECK(g.num_vertices() == 3);
    CHECK(g.num_edges() == 2);
    CHECK(g(0,1) == 3);
    CHECK(g(1,0) == 3);
    CHECK(g(1,2) == 5);
}

TEST_CASE("reading a directed graph") {
    Graph g;
    std::string str = "H 3 2 2\nE 0 1 3\nE 1 2 5\n";
    std::stringstream(str) >> g;
    CHECK(g.num_vertices() == 3);
    CHECK(g.num_edges() == 2);
    CHECK(g(0,1) == 3);
    CHECK(g(1,0) == Graph::WEIGHT_INFTY);
    CHECK(g(1,2) == 5);
}

TEST_CASE("read-write an undirected graph") {
    Graph g(3, false);
    g.set(0, 1, 4);
    g.set(1, 0, 5);
    std::stringstream out;
    out << g;
    Graph g2;
    out >> g2;
    CHECK(g.is_directed() == g2.is_directed());
    CHECK(g.num_edges() == g2.num_edges());
    CHECK(g.num_vertices() == g.num_vertices());
    CHECK(g(0,1) == g2(0,1));
    CHECK(g(1,0) == g2(1,0));
}

TEST_CASE("read-write a directed graph") {
    Graph g(3, true);
    g.set(0, 1, 4);
    g.set(1, 0, 5);
    std::stringstream out;
    out << g;
    Graph g2;
    out >> g2;
    CHECK(g.is_directed() == g2.is_directed());
    CHECK(g.num_edges() == g2.num_edges());
    CHECK(g.num_vertices() == g.num_vertices());
    CHECK(g(0,1) == g2(0,1));
    CHECK(g(1,0) == g2(1,0));
}