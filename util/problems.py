#
# Utility to calculate and pretty-print problem statistics
#

from collections import namedtuple
import sys


Problem = namedtuple('Problem', ['vertices', 'densities'])
Problem.__doc__ = '''A simple container for problem definitions'''


class ProblemDocumenter:
    '''Compute and print problem parameters for use in the report.'''

    def __init__(self, problem_list):
        self.problem_list = problem_list

    def cuda_memcost(self, V, E):
        '''Compute the memory requirements, in bytes, for the CUDA solution.

        Allocations as found in cuda/single-kernel
        '''
        # inbound_vertices: uint2 * 2 * V
        vcost = 4 * 2 * V
        # outbound_vertices: uint2 * 2 * E
        ecost = 4 * 2 * E
        return vcost + ecost

    def thrust_memcost(self, V, E):
        '''Compute the memory requirements, in bytes, for the CUDA Thrust solution.

        Allocations as found in thrust/
        '''
        # target: uint32_t * 2V, weight: int32_t * 2V (assuming undirected!)
        vcost = (4 * 2 * V) + (4 * 2 * V)
        # num_edges: uint32_t * E, idx_edges: uint32_t * E
        ecost = (4 * E) + (4 * E)
        return vcost + ecost

    def to_stdout(self):
        '''Pretty-print the problems to the console.'''
        for p in self.problem_list:
            V = float(p.vertices)
            max_E = V*(V-1)/2
            print(f'|V| = {V:.0}, max_edges = {max_E:.0}')
            print('   density      |E|      CUDA GiB      Thrust GiB')
            for d in p.densities:
                E = round(d*max_E, 0)
                # 2**32 is GiB
                cuda = self.cuda_memcost(V, E) / 2.0**32
                thrust = self.thrust_memcost(V, E) / 2.0**32
                print(f'    {d:<7}     {E:.0}  {cuda:>10.3f} {thrust:>10.3f}')
            print()

    def to_latex(self):
        '''Print the problems as a latex table.'''
        pass

    def to_cpp(self, numReplica=3, cntRuns=1):
        '''Print the problems as a latex table.'''
        for p in self.problem_list:
            V = p.vertices
            for d in p.densities:
                print(f'runParamSet(std::cout, {V}, MAX_INT32-1, {d}, {numReplica}, {cntRuns}')


if __name__ == '__main__':
    problem_list = [
        Problem(vertices=10**3, densities=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
        Problem(vertices=10**4, densities=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
        Problem(vertices=10**5, densities=[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]),
        Problem(vertices=10**6, densities=[0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004]),
    ]

    pd = ProblemDocumenter(problem_list)
    if len(sys.argv) > 1 and sys.argv[1] == 'cpp':
        pd.to_cpp()
    else:
        pd.to_stdout()
