#include "graph.hpp"
#include "matrix_graph.hpp"
#include "generator.hpp"

int main(int argc, const char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <vertices> <density> <min-weight> <max-weight>" << std::endl;
        return 1;
    }
    uint32_t num_vertices = atoi(argv[1]);
    float density = atof(argv[2]);
    int32_t min_weight = atoi(argv[3]);
    int32_t max_weight = atoi(argv[4]);
    MatrixGraph aGraph;
    generator(aGraph, num_vertices, min_weight, max_weight, density, false);
    // write to stdout
    std::cout << aGraph;
}