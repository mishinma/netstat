import os
import pytest
import numpy as np

from numpy.testing import assert_equal
from netstat.clean import clean_file
from netstat.polygon import load_graph_data, build_graph


@pytest.fixture(scope='session')
def network_fname(datadir):
    return os.path.join(datadir, 'network.txt')


@pytest.fixture(scope='session')
def network_clean_expected_fname(datadir):
    return os.path.join(datadir, 'network-clean-expected.txt')


@pytest.fixture(scope='session')
def network_clean_test_fname(datatmpdir):
    fname = os.path.join(datatmpdir, 'network-clean-test.txt')
    f = open(fname, 'w')
    f.close()
    return fname


@pytest.fixture(scope='session')
def graph_data(network_clean_expected_fname):
    return load_graph_data(network_clean_expected_fname)


@pytest.fixture(scope='session')
def num_nodes(graph_data):
    return graph_data[0]


@pytest.fixture(scope='session')
def num_edges(graph_data):
    return graph_data[1]


@pytest.fixture(scope='session')
def edges(graph_data):
    return graph_data[2]


@pytest.fixture(scope='session')
def graph_expected(datadir):
    arr_name = os.path.join(datadir, 'graph.npy')
    return np.load(arr_name)


def test_clean_file(network_fname, network_clean_test_fname, network_clean_expected_fname):
    clean_file(network_fname, network_clean_test_fname)
    with open(network_clean_test_fname, 'r') as f_test:
        with open(network_clean_expected_fname, 'r') as f_exp:
            assert f_test.read() == f_exp.read()


def test_build_graph(num_nodes, num_edges, edges, graph_expected):
    graph = build_graph(num_nodes, num_edges, edges)
    assert_equal(graph.toarray(), graph_expected)
