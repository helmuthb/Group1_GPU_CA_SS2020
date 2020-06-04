import networkx as nx
import sys


def print_mst(filename):
    G = nx.Graph()
    with open(sys.argv[1]) as fo:
        # Skip the header
        next(fo)
        for line in fo:
            _, x, y, w = line.split()
            G.add_edge(int(x), int(y), weight=int(w))
    T = nx.minimum_spanning_tree(G)
    print(T.size(weight='weight'))


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Usage: python3 minweight.py <file>')
        sys.exit(1)
    print_mst(sys.argv[1])
