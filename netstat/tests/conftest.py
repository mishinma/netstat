import os
import shutil
import pytest
import netstat


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
