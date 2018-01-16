"""Test utils module."""
import numpy as np
from numpy.testing import assert_array_equal, assert_equal

from regain import utils


def test_flatten():
    """Test flatten function."""
    a = [[1, 2], [3, 4], [5]]
    assert_array_equal(utils.flatten(a), np.arange(1, 6))


def test_upper_to_full():
    """Test upper_to_full function."""
    a = np.arange(9).reshape(3, 3)
    a += a.T
    upper = a[np.triu_indices(3)]
    assert_array_equal(utils.upper_to_full(upper), a)


def test_error_rank():
    """Test error_rank function."""
    a = np.arange(27).reshape(3, 3, 3)
    a += a.T
    assert_equal(utils.error_rank(a, a), 0)


def test_error_norm():
    """Test error_norm function."""
    a = np.arange(9).reshape(3, 3)
    a += a.T
    assert_equal(utils.error_norm(a, a), 0)


def test_error_norm_time():
    """Test error_norm_time function."""
    a = np.arange(27).reshape(3, 3, 3)
    a += a.T
    assert_equal(utils.error_norm_time(a, a), 0)
