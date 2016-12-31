import pytest
import numpy as np

from numpy.testing import assert_equal
from netstat.anf import least_zero_bit


@pytest.mark.parametrize("arr, least_zero_bit_exp", [
    (np.array([0, 0, 0]), np.array([0, 0, 0])),
    (np.array([0, 1, 3]), np.array([0, 1, 2])),
    (np.array([3, 7, 6]), np.array([2, 3, 0])),
    (np.array([[0, 1, 2], [3, 4, 5]]),
     np.array([[0, 1, 0], [2, 0, 1]]))
])
def test_mean(arr, least_zero_bit_exp):
    lzb = least_zero_bit(arr)
    assert_equal(lzb, least_zero_bit_exp)