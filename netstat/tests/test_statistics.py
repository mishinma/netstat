import pytest
import numpy as np

from numpy.testing import assert_almost_equal
from netstat.statistics import normalize, mean, median, effective


@pytest.mark.parametrize("arr, mean_exp", [
    (np.array([0, 1, 2, 1]), 2.0),
    (np.array([0, 2, 4, 3, 1, 1]), 2.545)
])
def test_mean(arr, mean_exp):
    arr_norm = normalize(arr)
    assert_almost_equal(mean(arr_norm), mean_exp, decimal=2)


@pytest.mark.parametrize("arr, median_exp", [
    (np.array([0, 1, 2, 1]), 2),
    (np.array([0, 1, 1, 1]), 2),
    (np.array([0, 1, 0, 1]), 1),
    (np.array([0, 2, 4, 3, 1, 1]), 2)
])
def test_median(arr, median_exp):
    arr_norm = normalize(arr)
    assert median(arr_norm) == median_exp


@pytest.mark.parametrize("arr, effective_exp", [
    (np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 9),
    (np.array([0, 1, 3, 5, 3, 4, 2, 1, 1, 0, 1]), 7),
    (np.array([0, 1, 3, 5, 3, 3, 2, 1, 1, 1]), 7)
])
def test_effective(arr, effective_exp):
    arr_norm = normalize(arr)
    assert effective(arr_norm) == effective_exp