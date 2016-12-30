import numpy as np

from scipy import stats

P = .5


def binrepr(arr):
    """ Helper function for printing arrays in binary repr"""
    return [np.binary_repr(_) for _ in arr]


def set_bits(n, l):
    """
    Create an approximate M(x,O)

    :param n: number of nodes
    :param l: length of bitstring
    :return: M(x,0) represented as np.array of ints
    """

    # Create a custom pmf for given l
    xk = np.arange(l + 1)  # value l means no bit set
    pk = [P ** (i + 1) for i in range(l)]
    # The prob. of not setting any bit is equal
    # to the prob. of setting the last bit
    pk.append(pk[-1])
    geom2 = stats.rv_discrete(name='geom2', values=(xk, pk))
    bits = geom2.rvs(size=n)  # Sample n values from the pmf

    # Create M(x,0)
    M = np.ones(n, dtype=np.int32)
    M = np.left_shift(M, bits)
    mask = int('1' * l, 2)
    M = np.bitwise_and(M, mask)

    return M


