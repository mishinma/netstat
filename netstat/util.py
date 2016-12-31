""" Module for helper functions """
import time
import numpy as np
import pandas as pd

from math import ceil, log
from scipy.sparse import csr_matrix, csgraph, stats


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


def largest_connected_component(graph, connection='strong'):
    """ Get the largest connected component and return it as a submatrix """

    print "Finding the largest connected component..."

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


def least_zero_bit(arr):
    res = np.empty(arr.shape, dtype=np.int32)
    i = 0
    set_mask = np.zeros(arr.shape, dtype=bool)
    while True:
        lzb = np.bitwise_and(arr, 1) == 0
        res[np.logical_and(lzb, np.logical_not(set_mask))] = i
        arr = np.right_shift(arr, 1)
        set_mask = np.logical_or(set_mask, lzb)
        i += 1
        if np.all(set_mask):
            break

    return res


def set_bits(n, r, k):
    """
    Create an approximate M(x,O)

    :param n: number of nodes
    :param r: some small constant
    :param k: number of approximations
    :return: M(x,0) represented as np.array of ints
    """
    # Calculate the length of bitstring
    l = int(ceil(log(n, 2))) + r
    # Create a custom pmf for given l
    xk = np.arange(l + 1)  # value l means no bit set
    pk = [P ** (i + 1) for i in range(l)]
    # The prob. of not setting any bit is equal
    # to the prob. of setting the last bit
    pk.append(pk[-1])
    geom2 = stats.rv_discrete(name='geom2', values=(xk, pk))
    bits = geom2.rvs(size=k*n)  # Sample k*n values from the pmf
    bits.shape = (n,k)

    # Create M(x,0)
    M = np.ones((n,k), dtype=np.int32)
    M = np.left_shift(M, bits)
    mask = int('1' * l, 2)
    M = np.bitwise_and(M, mask)

    return M