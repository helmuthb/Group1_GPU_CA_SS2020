#!/usr/bin/env python

import csv
import os

import matplotlib.pyplot as plt


class ResultsDict(dict):
    """Subclass dict so that we can add attributes."""
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


def parse(filename):
    """Reads a results file and returns data structure of the form:

          result['implementation']['vertices']['density'] = avg_runtime

    The data structure has two additional attributes:

          result.vertices  = [sorted list of seen vertices]
          result.densities = [sorted list of seen densities]

    runs, seed, weight_range are ignored.
    """
    results = ResultsDict()
    seen_vertices = set()
    seen_densities = set()

    # First, collect all results
    with open(filename) as fo:
        reader = csv.DictReader(fo)

        for line in reader:
            impl = line['implementation']
            vert = int(line['vertices'])
            dens = float(line['density'])

            # Create the chain of dicts
            if impl not in results:
                results[impl] = {}
            if vert not in results[impl]:
                results[impl][vert] = {}
            if dens not in results[impl][vert]:
                results[impl][vert][dens] = []

            # Store the individual result
            results[impl][vert][dens].append(float(line['runtime']))

        for impl in results:
            for vert in results[impl]:
                seen_vertices.add(vert)
                for dens in results[impl][vert]:
                    results[impl][vert][dens] = ( sum(results[impl][vert][dens]) /
                                                  len(results[impl][vert][dens]) )
                    seen_densities.add(dens)

        results.vertices = sorted(seen_vertices)
        results.densities = sorted(seen_densities)
        return results


def results_by_density(parsed):
    """Converts parse() results to a dict of lists of densities.

          result['implementation'] = [list of avg_runtimes]

    The data structure has two additional attributes:

          result.vertex    = vertex
          result.densities = [sorted list of seen densities]
    """
    results = ResultsDict()
    results.densities = parsed.densities
    # We assume that vertex is constant (only one value exists)
    results.vertex = parsed.vertices[0]
    for impl in parsed:
        results[impl] = []
        for density in parsed.densities:
            results[impl].append(parsed[impl][results.vertex][density])

    return results


def results_by_vertex(parsed):
    """Converts parse() results to a dict of lists of vertices.

          result['implementation'] = [list of avg_runtimes]

    The data structure has two additional attributes:

          result.vertices = [sorted list of seen vertices]
          result.density  = density
    """
    results = ResultsDict()
    results.vertices = parsed.vertices
    # We assume that density is constant (only one value exists)
    results.density = parsed.densities[0]
    for impl in parsed:
        results[impl] = []
        for vertex in parsed.vertices:
            results[impl].append(parsed[impl][vertex][results.density])

    return results


if __name__ == '__main__':
    if not os.path.exists('results_density.txt'):
        raise FileNotFoundError('Missing input file results_density.txt')
    if not os.path.exists('results_vertices.txt'):
        raise FileNotFoundError('Missing input file results_vertices.txt')

    density_results = results_by_density(parse('results_density.txt'))
    for impl in density_results:
        # CPU is really slow and warps the plot
        if impl == 'cpu_l':
            continue
        xs = density_results.densities
        ys = density_results[impl]
        plt.plot(xs, ys, label=impl)
    plt.title(f'Performance by Density\nV={density_results.vertex}')
    plt.xlabel('Density')
    plt.ylabel('Runtime (ms)')
    plt.legend(loc='best')
    plt.savefig('results_density.png')
    plt.close()

    vertex_results = results_by_vertex(parse('results_vertices.txt'))
    for impl in vertex_results:
        # CPU is really slow and warps the plot
        if impl == 'cpu_l':
            continue
        xs = vertex_results.vertices
        ys = vertex_results[impl]
        plt.plot(xs, ys, label=impl)
    plt.title(f'Performance by Density\nV={vertex_results.density}')
    plt.xlabel('Vertices')
    plt.ylabel('Runtime (ms)')
    plt.legend(loc='best')
    plt.savefig('results_vertices.png')
    plt.close()
