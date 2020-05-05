#include "graph.hpp"
#include "generator.hpp"

int main(int argc, const char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <vertices> <density> <min-weight> <max-weight>" << std::endl;
        return 1;
    }
    int num_vertices = atoi(argv[1]);
    float density = atof(argv[2]);
    int min_weight = atoi(argv[3]);
    int max_weight = atoi(argv[4]);
    Graph aGraph = generator(num_vertices, min_weight, max_weight, density, false);
    // write to stdout
    std::cout << aGraph;
}