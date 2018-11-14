"""Test utils module."""
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from regain import prox


def test_soft_thresholding():
    """Test soft_thresholding function."""
    # array
    array = np.arange(3)
    output = np.array([0, 0.5, 1.5])
    assert_array_equal(prox.soft_thresholding(array, .5), output)

    # matrix
    array = np.arange(9).reshape(3, 3)
    output = np.array([[0, 0, 1], [2, 3, 4], [5, 6, 7]])
    assert_array_equal(prox.soft_thresholding(array, 1), output)

    # tensor
    array = np.arange(27).reshape(3, 3, 3)
    output = array - 1
    output[0, 0, 0] = 0
    assert_array_equal(prox.soft_thresholding(array, 1), output)

    # tensor, lamda is a matrix
    array = np.arange(27).reshape(3, 3, 3)
    output = array - 1
    output[0, 0, 0] = 0
    output[1] -= 1
    output[2] -= 2
    assert_array_equal(
        prox.soft_thresholding(array,
                               np.arange(1, 4)[:, None, None]), output)


def test_soft_thresholding_od():
    """Test soft_thresholding_od function."""
    # matrix OD
    array = np.arange(9).reshape(3, 3)
    output = np.array([[0, 0, 1], [2, 4, 4], [5, 6, 8]])
    assert_array_equal(prox.soft_thresholding_od(array, 1), output)

    # tensor OD
    array = np.arange(27).reshape(3, 3, 3)
    output = np.array(
        [
            [[0, 0, 1], [2, 4, 4],
             [5, 6, 8]], [[9, 9, 10], [11, 13, 13], [14, 15, 17]],
            [[18, 18, 19], [20, 22, 22], [23, 24, 26]]
        ])
    assert_array_equal(prox.soft_thresholding_od(array, 1), output)

    # tensor OD, lamda is a list
    array = np.arange(27).reshape(3, 3, 3)
    output = np.array(
        [
            [[0, 0, 1], [2, 4, 4],
             [5, 6, 8]], [[9, 8, 9], [10, 13, 12], [13, 14, 17]],
            [[18, 16, 17], [18, 22, 20], [21, 22, 26]]
        ])

    assert_array_equal(
        prox.soft_thresholding_od(array, np.arange(1, 4)), output)


def test_soft_thresholding_vector():
    """Test soft_thresholding_vector function."""
    # array
    array = np.arange(3)
    output = np.array([0, 0.7763932, 1.5527864])
    assert_array_almost_equal(prox.soft_thresholding_vector(array, .5), output)

    # matrix
    array = np.arange(9).reshape(3, 3)
    output = np.array(
        [
            [0., 0.87690851, 1.79260966], [2.5527864, 3.50763404, 4.48152415],
            [5.10557281, 6.13835956, 7.17043864]
        ])
    assert_array_almost_equal(
        prox.blockwise_soft_thresholding(array, 1), output)

    arr = array + array.T
    assert_array_equal(
        prox.blockwise_soft_thresholding(arr, 1),
        prox.blockwise_soft_thresholding_symmetric(arr, 1))

    # tensor
    arr3 = np.array([arr] * 2)
    output = np.array(
        [
            [0., 3.73273876, 7.62860932], [3.5527864, 7.46547752, 11.44291399],
            [7.10557281, 11.19821627, 15.25721865]
        ])
    out = np.array([output] * 2)
    assert_array_almost_equal(prox.blockwise_soft_thresholding(arr3, 1), out)

    assert_array_almost_equal(
        prox.blockwise_soft_thresholding_symmetric(arr3, 1), out)
