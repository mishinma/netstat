import os
import shutil
import pytest
import netstat

from netstat.polygon import load_graph_data

@pytest.fixture(scope='session')
def datadir(request):
    dirname = os.path.dirname(netstat.__file__)
    return os.path.join(dirname, 'tests', 'data')


@pytest.fixture(scope='session')
def datatmpdir(datadir, request):
    dir = os.path.join(datadir, 'tmp')
    if not os.path.exists(dir):
        os.makedirs(dir)

    def fin():
        shutil.rmtree(dir)

    request.addfinalizer(fin)
    return dir


@pytest.fixture(scope='session')
def network_fname(datadir):
    return os.path.join(datadir, 'network.txt')


@pytest.fixture(scope='session')
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
