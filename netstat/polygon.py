import os
import sys
import h5py
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph


# def timer(f):
#     """A decorator for timing code"""
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         res = f(*args, **kwargs)
#         elapsed = (time.time() - start_time)
#         print "--- {} s ---".format(elapsed)
#         return res
#     return wrapper


def load_graph_data(fname):

    print "Loading the graph data..."

    with open(fname, 'r') as f:
        num_nodes, num_edges = map(int, next(f).split())

    edges = pd.read_csv(fname, sep=' ', names=['FromNodeId', 'ToNodeId'],
                             dtype={'FromNodeId': np.int32, 'ToNodeId': np.int32}, skiprows=1)

    return num_nodes, num_edges, edges


def build_graph(num_nodes, num_edges, edges):

    print "Building the adjacency matrix..."

    edges = edges.values  # Access values as np array

    row = edges[:, 0]
    col = edges[:, 1]

    row_counts = np.bincount(row, minlength=num_nodes)
    indptr = np.hstack([0, np.cumsum(row_counts)])

    data = np.full(shape=num_edges, fill_value=1, dtype=np.uint8)

    graph = csr_matrix((data, col, indptr), shape=(num_nodes, num_nodes))

    return graph


def get_lcc(graph, directed=True, connection='strong'):
    """ Get the largest connected component and return it as a submatrix"""

    print "Finding the largest connected component..."

    num_components, labels = csgraph.connected_components(graph, directed=directed, connection=connection)
    lcc_label = np.argmax(np.bincount(labels))
    indices = np.nonzero(labels == lcc_label)[0]

    # Get the submatrix for the LCC
    # Before slicing the columns convert to the CSC sparse format
    lcc = graph[indices, :].tocsc()[:, indices].tocsr()

    return lcc


def breadth_first_search(graph, i_start, directed=True):

    nodes, predecessors = csgraph.breadth_first_order(graph, i_start=i_start, directed=directed)

    dist = np.empty(nodes.shape, dtype=np.int32)
    dist[i_start] = 0

    for i in nodes[1:]:
        dist[i] = dist[predecessors[i]] + 1

    dist_distr = np.bincount(dist)[1:]  # Don't take zeros into account
    return dist_distr


def get_distance_distribution(graph, nodes, directed=True):

    print "Computing the distance distribution..."

    dist_distr = np.zeros(shape=10, dtype=np.int32)

    for node in nodes:
        node_dist_distr = breadth_first_search(graph, i_start=node, directed=directed)

        if dist_distr.size < node_dist_distr.size:
            dist_distr.resize(node_dist_distr.shape)

        dist_distr[:node_dist_distr.size] += node_dist_distr

    return dist_distr


if __name__ == '__main__':
    fname = sys.argv[1]
    # store = sys.argv[2]
    network_name = os.path.splitext(os.path.basename(fname))[0]
    if network_name.endswith('-clean'):
        network_name = network_name[:-6]
    print "Network name: " + network_name
    start_time = time.time()
    num_nodes, num_edges, graph_data = load_graph_data(fname)
    graph = build_graph(num_nodes, num_edges, graph_data)
    lcc = get_lcc(graph, directed=True, connection='strong')
    all_nodes = np.arange(lcc.shape[0])
    dist_distr = get_distance_distribution(lcc, all_nodes, directed=True)
    elapsed = (time.time() - start_time)
    print "--- {} s ---".format(elapsed)
    print "Done"