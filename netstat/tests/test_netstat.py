import os
import pytest
import numpy as np

from numpy.testing import assert_equal
from netstat.clean import clean_file
from netstat.exact import exact_distance_distribution
from netstat.util import load_graph_data, build_graph, largest_connected_component


@pytest.fixture(scope='module')
def network_fname(datadir):
    return os.path.join(datadir, 'network.txt')


@pytest.fixture(scope='module')
def network_clean_expected_fname(datadir):
    return os.path.join(datadir, 'network-clean-expected.txt')


@pytest.fixture(scope='module')
def network_clean_test_fname(datatmpdir):
    fname = os.path.join(datatmpdir, 'network-clean-test.txt')
    f = open(fname, 'w')
    f.close()
    return fname


@pytest.fixture(scope='module')
def graph_data(network_clean_expected_fname):
    return load_graph_data(network_clean_expected_fname)


@pytest.fixture(scope='module')
def num_nodes(graph_data):
    return graph_data[0]


@pytest.fixture(scope='module')
def num_edges(graph_data):
    return graph_data[1]


@pytest.fixture(scope='module')
def edges(graph_data):
    return graph_data[2]


@pytest.fixture(scope='module')
def graph(num_nodes, num_edges, edges):
    return build_graph(num_nodes, num_edges, edges)


@pytest.fixture(scope='module')
def lscc(graph):
    return largest_connected_component(graph, connection='strong')


@pytest.fixture(scope='module')
def lwcc(graph):
    return largest_connected_component(graph, connection='weak')


def test_clean_file(network_fname, network_clean_test_fname, network_clean_expected_fname):
    clean_file(network_fname, network_clean_test_fname)
    with open(network_clean_test_fname, 'r') as f_test:
        with open(network_clean_expected_fname, 'r') as f_exp:
            assert f_test.read() == f_exp.read()


def test_build_graph(graph):
    graph_expected = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])
    assert_equal(graph.toarray(), graph_expected)


def test_largest_connected_component_strong(lscc):
    lscc_expected = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    assert_equal(lscc.toarray(), lscc_expected)


def test_largest_connected_component_weak(lwcc):
    lwcc_expected = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]
    ])
    assert_equal(lwcc.toarray(), lwcc_expected)


def test_exact_distance_distribution_directed(lscc):
    dist_distr = exact_distance_distribution(lscc, directed=True)
    dist_distr_expected = np.array([3, 3])
    assert_equal(dist_distr[1:3], dist_distr_expected)


def test_exact_distance_distribution_undirected(lwcc):
    dist_distr = exact_distance_distribution(lwcc, directed=False)
    dist_distr_expected = np.array([4, 2])
    assert_equal(dist_distr[1:3], dist_distr_expected)
