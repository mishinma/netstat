import pytest
import numpy as np

from numpy.testing import assert_equal
from netstat.graph import Graph, breadth_first_search


@pytest.fixture(scope='module')
def graph(network_clean_expected_fname):
    return Graph.loadtxt(network_clean_expected_fname)


def test_graph_init(graph, num_nodes, edges):
    import pdb; pdb.set_trace()
    assert graph.indptr.shape[0] == num_nodes + 1
    assert graph.indices.shape[0] == edges.shape[0]


def test_graph_adjacent(graph):
    assert_equal(graph.adjacent(3), np.array([1,2]))
    assert graph.adjacent(2).size == 0


@pytest.mark.parametrize("start, dist_expected",[
    (0, np.array([0, 1, 2, np.inf, np.inf, np.inf])),
    (3, np.array([np.inf, 1, 1, 0, np.inf, np.inf])),
    (4, np.array([np.inf, np.inf, np.inf, np.inf, 0, 1])),
    (5, np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 0]))
])
def test_breadth_first_search(graph, start, dist_expected):
    dist = breadth_first_search(graph, start)
    assert_equal(dist, dist_expected)

