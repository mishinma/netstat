""" Module for exact computation"""

import numpy as np

from netstat.graph import breadth_first_search

MAX_NUM_DIST = 101


def exact_distance_distribution(graph, nodes=None, directed=True):
    """ Get the distance distribution of a graph

        Returns the distribution in the following format:

            ind:  x 1 2 3 4 5 6 ... 100 (or maximum dist)
            val:  0 2 4 2 1 0 1 ... 0
    """

    if nodes is None:
        nodes = np.arange(graph.shape[0])  # Run for all nodes

    print "Computing the distance distribution..."

    # Assume there will be at most 100 different distances in the distribution
    # The actual length is 101 so that indices correspond to distances' values.
    dist_distr = np.zeros(shape=MAX_NUM_DIST, dtype=np.int64)

    for node in nodes:
        # Get the distance distribution from `node`
        _, distances = breadth_first_search(graph, i_start=node, directed=directed)
        node_dist_distr = np.bincount(distances, minlength=MAX_NUM_DIST)

        if dist_distr.size < node_dist_distr.size:
            dist_distr.resize(node_dist_distr.size)

        # Cumulatively add to other values
        dist_distr[:node_dist_distr.size] += node_dist_distr

    dist_distr[0] = 0  # Set to 0 so it doesn't affect the distribution

    return dist_distr

