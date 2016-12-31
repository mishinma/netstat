import sys
import os
import time
import numpy as np

from math import log, ceil
from itertools import izip
from scipy import stats, sparse
from netstat.polygon import build_graph, load_graph_data, get_lcc
from netstat.statistics import compute_statistics


P = .5
PHI = .77351  # Proportionality constant


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


# Naive version - slow
# def _lzb(x):
#         i = 0
#         while x & 1:
#             x >>= 1
#             i += 1
#         return i
#
# def least_zero_bit(arr):
#
#     res = np.empty_like(arr)
#
#     if res.ndim == 1:
#         for i in range(res.shape[0]):
#             res[i] = _lzb(arr[i])
#     else:
#         for i in range(res.shape[0]):
#             for j in range(res.shape[1]):
#                 res[i, j] = _lzb(arr[i,j])
#
#     return res


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


def anf0(graph, k, r, num_dist=None, nodes=None, directed=True):

    if nodes is None:
        nodes = np.arange(graph.shape[0])  # Run for all nodes

    n = nodes.shape[0]

    if num_dist is None:
        num_dist = n

    # Mcur = np.vstack(set_bits(n, l, r) for _ in range(k)).T

    Mcur = set_bits(n, r, k)
    graph = graph.tocoo()  # Convert to COOrdinate format

    approx_N = np.empty(num_dist+1)
    approx_N[0] = n

    if directed:
        _anf0_directed(graph, num_dist, Mcur, approx_N)
    else:
        _anf0_undirected(graph, num_dist, Mcur, approx_N)

    approx_N[1:] -= approx_N[:-1].copy()  # "Reverse" cumsum
    approx_N[0] = 0

    return approx_N


def _anf0_directed(graph, num_dist, Mcur, approx_N):
    for h in range(1, num_dist+1):
        Mlast = Mcur.copy()

        # Iterate through all edges
        for u, v in izip(graph.row, graph.col):

            Mcur[u, :] = np.bitwise_or(Mcur[u, :], Mlast[v, :])

        approx_IN = np.exp2(np.mean(least_zero_bit(Mcur), axis=1))/PHI
        approx_N[h] = np.sum(approx_IN)


def _anf0_undirected(graph, num_dist, Mcur, approx_N):
    for h in range(1, num_dist+1):
        Mlast = Mcur.copy()

        # Iterate through all edges
        for u, v in izip(graph.row, graph.col):

            Mcur[u, :] = np.bitwise_or(Mcur[u, :], Mlast[v, :])
            Mcur[v, :] = np.bitwise_or(Mcur[v, :], Mlast[u, :])

        approx_IN = np.exp2(np.mean(least_zero_bit(Mcur), axis=1))/PHI
        approx_N[h] = np.sum(approx_IN)


if __name__ == '__main__':
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={'float_kind': float_formatter})
    fname = sys.argv[1]  # The name of a clean file as the first arg
    try:
        sys.argv[2] == 'undir'
    except:
        directed, connection = True, 'strong'
    else:
        directed, connection = False, 'weak'
    # store = sys.argv[2]  # The name of an HDF store as the second arg
    network_name = os.path.splitext(os.path.basename(fname))[0]
    if network_name.endswith('-clean'):
        network_name = network_name[:-6]
    print "Network name: " + network_name
    start_time = time.time()
    num_nodes, num_edges, graph_data = load_graph_data(fname)
    graph = build_graph(num_nodes, num_edges, graph_data)
    lcc = get_lcc(graph, connection=connection)
    print "Nodes: {}, Edges {}".format(lcc.shape[0], lcc.data.shape[0])
    N = anf0(lcc, k=50, r=1, num_dist=10, directed=directed)
    print N
    mn, mdn, diam, eff_diam = compute_statistics(N)
    print "Mean {}".format(mn)
    print "Median {}".format(mdn)
    print "Diameter {}".format(diam)
    print "Eff diameter {}".format(eff_diam)
    elapsed = (time.time() - start_time)
    print "--- {} m ---".format(elapsed / 60)
    print "Done"

