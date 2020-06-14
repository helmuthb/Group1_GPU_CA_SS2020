import sys

import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(filename):
    G = nx.Graph()
    with open(sys.argv[1]) as fo:
        # Skip the header
        next(fo)
        for line in fo:
            _, x, y, w = line.split()
            G.add_edge(int(x), int(y), weight=int(w))

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.savefig(filename + '.png')


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Usage: python3 draw-graph.py <file>')
        sys.exit(1)
    draw_graph(sys.argv[1])
