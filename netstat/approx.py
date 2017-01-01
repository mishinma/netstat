import numpy as np

from itertools import izip
from math import ceil, log
from scipy import stats
from netstat.graph import breadth_first_search
from netstat.exact import exact_distance_distribution, MAX_NUM_DIST


P = .5
PHI = .77351  # Proportionality constant


def sample_random_pairs(graph, k, directed):
    samples = np.random.choice(np.arange(graph.shape[0]), (k, 2))
    dist_distr = np.zeros(shape=MAX_NUM_DIST, dtype=np.int64)
    for pair in samples:
        source_dist = breadth_first_search(graph, i_start=pair[0], directed=directed)
        dist = source_dist[1][pair[1]]
        if dist > dist_distr.size:
            dist_distr.resize(dist+1)
        dist_distr[dist] += 1
    return dist_distr


def sample_random_sources(graph, k, directed):
    samples = np.random.choice(np.arange(graph.shape[0]), k)
    return exact_distance_distribution(graph, samples, directed)


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