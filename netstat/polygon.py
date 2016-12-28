import os
import sys
import h5py
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph


def load_graph_data(fname):

    with open(fname, 'r') as f:
        num_nodes, num_edges = map(int, next(f).split())

    edges = pd.read_csv(fname, sep=' ', names=['FromNodeId', 'ToNodeId'],
                             dtype={'FromNodeId': np.int32, 'ToNodeId': np.int32}, skiprows=1)

    return num_nodes, num_edges, edges


def build_graph(num_nodes, num_edges, edges):

    edges = edges.values  # Access values as np array

    row = edges[:, 0]
    col = edges[:, 1]

    row_counts = np.bincount(row, minlength=num_nodes)
    indptr = np.hstack([0, np.cumsum(row_counts)])

    data = np.full(shape=num_edges, fill_value=1, dtype=np.uint8)

    graph = csr_matrix((data, col, indptr), shape=(num_nodes, num_nodes))

    return graph


def get_lcc(graph, directed=False, connection='weak'):
    """ Get the largest connected component and return it as a submatrix"""

    num_components, labels = csgraph.connected_components(graph, directed=directed, connection=connection)
    lcc_label = np.argmax(np.bincount(labels))
    indices = np.nonzero(labels == lcc_label)[0]

    # Get the submatrix for the LCC
    # Before slicing the columns convert to the CSC sparse format
    lcc = graph[indices, :].tocsc()[:, indices].tocsr()

    return lcc


def breadth_first_dist(graph, i_start, directed=False):
    nodes, predecessors = csgraph.breadth_first_order(graph, i_start=i_start, directed=directed)
    dist = np.full(nodes.shape, -9999)
    dist[i_start] = 0


if __name__ == '__main__':
    fname = sys.argv[1]
    # store = sys.argv[2]
    network_name = os.path.splitext(os.path.basename(fname))[0]
    if network_name.endswith('-clean'):
        network_name = network_name[:-6]
    print "Network name: " + network_name
    num_nodes, num_edges, graph_data = load_graph_data(fname)
    print "Building adjacency matrix..."
    graph = build_graph(num_nodes, num_edges, graph_data)
    print "Done"
    lcc = get_lcc(graph, directed=False, connection='weak')
    breadth_first_dist(lcc, 1, directed=False)
    # print "Running bfs..."
    # start_time = time.time()
    # # dist_matrix = calculate_dist_matrix(graph, store, network_name)
    # csgraph.breadth_first_order(graph, 0, directed=True)
    # elapsed = (time.time() - start_time)
    # print "Done"
    # print "--- {} seconds ---" .format(elapsed)
    # print "--- {} minutes ---" .format(elapsed / 60)




