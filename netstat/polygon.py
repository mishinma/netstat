import os
import sys
import h5py

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph


def load_graph_data(fname):

    with open(fname, 'r') as f:
        num_nodes, num_edges = map(int, next(f).split())

    edges = pd.read_csv(fname, sep=' ', names=['FromNodeId', 'ToNodeId'],
                             dtype={'FromNodeId': np.int64, 'ToNodeId': np.int64}, skiprows=1)

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


def calculate_dist_matrix(graph, store_name, network_name, directed=False, save=True):

    dist_matrix = csgraph.shortest_path(graph, method='D', directed=directed, unweighted=True)

    if save:
        with h5py.File(store_name) as f:
            network_name = network_name + '-dir' if directed else network_name
            dataset_name = 'shortest_paths/{}'.format(network_name)
            f.create_dataset(dataset_name, data=dist_matrix)

    return dist_matrix


def calculate_connected_components(graph, store_name, network_name, connection='weak', save=True):

    num_components, labels = csgraph.connected_components(graph, connection=connection)
    _, counts = np.unique(labels, return_counts=True)

    counts[::-1].sort()  # Sort in descending order

    print "Num edges in the largest CC: {}".format(counts[0])

    if save:
        with h5py.File(store_name) as f:
            network_name = network_name + '_' + connection
            dataset_name = 'connected_components/{}'.format(network_name)
            f.create_dataset(dataset_name, data=dist_matrix)

    return num_components, labels, counts


if __name__ == '__main__':
    fname = sys.argv[1]
    store = sys.argv[2]
    network_name = os.path.splitext(os.path.basename(fname))[0]
    if network_name.endswith('-clean'):
        network_name = network_name[-6:]
    print network_name
    num_nodes, num_edges, graph_data = load_graph_data(fname)
    graph = build_graph(num_nodes, num_edges, graph_data)
    dist_matrix = calculate_dist_matrix(graph, store, network_name)





