""" Module for helper functions """
import time
import numpy as np
# import pandas as pd

from math import ceil, log
from scipy.sparse import csr_matrix, csgraph


P = .5
SHEBANG = "#!clean"


class FileFormatError(Exception):
    pass


def load_graph_data(fname):
    """ Load graph data from a clean file """

    with open(fname, 'r') as f:
        first_line = next(f).strip()
        if first_line != SHEBANG:
            raise FileFormatError("The format of this file is wrong! Please provide option `-c`"
                                  " or use script `clean` to format the file.")
        num_nodes, num_edges = map(int, next(f).split())

    edges = np.loadtxt(fname, dtype=np.int32, skiprows=2)

    # Pandas's function is much faster
    # edges = pd.read_csv(fname, sep=' ', names=['FromNodeId', 'ToNodeId'],
    #                          dtype={'FromNodeId': np.int32, 'ToNodeId': np.int32}, skiprows=2)

    return num_nodes, num_edges, edges


def build_graph(num_nodes, num_edges, edges):
    """ Create a sparse CSR matrix representing the graph """

    # If `edges` is a DataFrame
    # edges = edges.values  # Access values as np array

    row = edges[:, 0].copy(order='C')
    col = edges[:, 1].copy(order='C')

    row_counts = np.bincount(row, minlength=num_nodes)
    indptr = np.hstack([0, np.cumsum(row_counts)])

    data = np.full(shape=num_edges, fill_value=1, dtype=np.uint8)

    graph = csr_matrix((data, col, indptr), shape=(num_nodes, num_nodes))

    return graph


def largest_connected_component(graph, connection='strong'):
    """ Get the largest connected component and return it as a submatrix """

    num_components, labels = csgraph.connected_components(graph, connection=connection)

    lcc_label = np.argmax(np.bincount(labels))  # Find the largest label
    indices = np.nonzero(labels == lcc_label)[0]  # Find the corresponding indices

    # Get the submatrix for the LCC
    # Before slicing the columns convert to the CSC sparse format
    lcc = graph[indices, :].tocsc()[:, indices].tocsr()

    return lcc


def timer(f):
    """A decorator for timing code"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = f(*args, **kwargs)
        elapsed = (time.time() - start_time)
        print "--- {} s ---".format(elapsed)
        return res
    return wrapper


def binrepr(arr, l=None):
    """ Helper function for printing arrays in binary repr"""

    if l is None:
        l = int(ceil(log(max(arr), 2)))

    res = np.empty_like(arr, dtype='S{}'.format(l))

    if res.ndim == 1:
        for i in range(res.shape[0]):
            res[i] = np.binary_repr(arr[i],  width=l)
    else:
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i, j] = np.binary_repr(arr[i,j],  width=l)

    return res


def print_statistics(mean, median, diam, eff_diam):
    print "Mean: {}".format(mean)
    print "Median: {}".format(median)
    print "Diameter: {}".format(diam)
    print "Eff diameter: {}".format(eff_diam)
