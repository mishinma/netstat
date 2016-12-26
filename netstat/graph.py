import numpy as np
from collections import deque


class Graph(object):

    @classmethod
    def loadtxt(cls, fname):

        # Read data from file
        graph_data = np.loadtxt(fname, dtype=np.int64)
        num_nodes, num_edges = graph_data[0]

        edges = graph_data[1:]
        row_counts = np.bincount(edges[:, 0], minlength=num_nodes)
        indptr = np.hstack([0, np.cumsum(row_counts)])
        indices = edges[:, 1]

        return cls(indices, indptr)

    def __init__(self, indices, indptr):
        self.indices = indices
        self.indptr = indptr
        self.num_nodes = indptr.size - 1

    def adjacent(self, i):
        return self.indices[self.indptr[i]: self.indptr[i+1]]


def breadth_first_search(graph, start):

    dist = np.empty(graph.num_nodes)
    dist.fill(np.inf)
    dist[start] = 0

    nodes_queue = deque([start])

    while nodes_queue:

        u = nodes_queue.popleft()
        for v in graph.adjacent(u):
            if np.isinf(dist[v]):
                nodes_queue.append(v)
                dist[v] = dist[u] + 1

    return dist
