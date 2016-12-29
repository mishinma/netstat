import os
import sys
import h5py
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph
from netstat.graph import breadth_first_search


def timer(f):
    """A decorator for timing code"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = f(*args, **kwargs)
        elapsed = (time.time() - start_time)
        print "--- {} s ---".format(elapsed)
        return res
    return wrapper


def load_graph_data(fname):
    """ Load graph data from a clean file """
    print "Loading the graph data..."

    with open(fname, 'r') as f:
        num_nodes, num_edges = map(int, next(f).split())

    edges = pd.read_csv(fname, sep=' ', names=['FromNodeId', 'ToNodeId'],
                             dtype={'FromNodeId': np.int32, 'ToNodeId': np.int32}, skiprows=1)

    return num_nodes, num_edges, edges


def build_graph(num_nodes, num_edges, edges):
    """ Create a sparse CSR matrix representing the graph """
    print "Building the adjacency matrix..."

    edges = edges.values  # Access values as np array

    row = edges[:, 0]
    col = edges[:, 1]

    row_counts = np.bincount(row, minlength=num_nodes)
    indptr = np.hstack([0, np.cumsum(row_counts)])

    data = np.full(shape=num_edges, fill_value=1, dtype=np.uint8)

    graph = csr_matrix((data, col, indptr), shape=(num_nodes, num_nodes))

    return graph


def get_lcc(graph, connection='strong'):
    """ Get the largest connected component and return it as a submatrix """

    print "Finding the largest connected component..."

    num_components, labels = csgraph.connected_components(graph, connection=connection)
    lcc_label = np.argmax(np.bincount(labels))  # Find the largest label
    indices = np.nonzero(labels == lcc_label)[0]  # Find the corresponding indices

    # Get the submatrix for the LCC
    # Before slicing the columns convert to the CSC sparse format
    lcc = graph[indices, :].tocsc()[:, indices].tocsr()

    return lcc


def get_distance_distribution(graph, nodes=None, directed=True, parallel=False):
    """ Get the distance distribution of a graph """

    if nodes is None:
        nodes = np.arange(graph.shape[0])  # Run for all nodes

    print "Computing the distance distribution..."

    #ToDo: not sure if that's reasonable
    # Assume there will be at most 100 different distances in the distribution
    # so that won't have to resize later
    dist_distr = np.zeros(shape=100, dtype=np.int32)

    for node in nodes:
        # print "--- node {} ---".format(node)
        # Get the distance distribution from `node`
        _, distances = breadth_first_search(graph, i_start=node, directed=directed)
        node_dist_distr = np.bincount(distances)[1:]  # Drop the zero distance

        if dist_distr.size < node_dist_distr.size:
            dist_distr.resize(node_dist_distr.shape)

        # Cumulatively add to other values
        dist_distr[:node_dist_distr.size] += node_dist_distr

    if not directed and not parallel:
        dist_distr /= 2  # because dist(u,v) = dist(v,u)

    return dist_distr


if __name__ == '__main__':
    fname = sys.argv[1]  # The name of a clean file as the first arg
    # store = sys.argv[2]  # The name of an HDF store as the second arg
    network_name = os.path.splitext(os.path.basename(fname))[0]
    if network_name.endswith('-clean'):
        network_name = network_name[:-6]
    print "Network name: " + network_name
    start_time = time.time()
    num_nodes, num_edges, graph_data = load_graph_data(fname)
    graph = build_graph(num_nodes, num_edges, graph_data)
    lcc = get_lcc(graph, connection='strong')
    dist_distr = get_distance_distribution(lcc, directed=False)

    # Store the distribution
    # with h5py.File(store) as f:
    #     f.create_dataset(network_name, data=dist_distr, dtype=np.int32)
    elapsed = (time.time() - start_time)
    print "--- {} m ---".format(elapsed / 60)
    print "Done"
