# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Test utils module."""
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from regain import prox


def test_soft_thresholding():
    """Test soft_thresholding function."""
    # array
    array = np.arange(3)
    output = np.array([0, 0.5, 1.5])
    assert_array_equal(prox.soft_thresholding(array, 0.5), output)

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
        prox.soft_thresholding(array, np.arange(1, 4)[:, None, None]), output
    )


def test_soft_thresholding_od():
    """Test soft_thresholding_od function."""
    # matrix OD
    array = np.arange(9).reshape(3, 3)
    output = np.array([[0, 0, 1], [2, 4, 4], [5, 6, 8]])
    assert_array_equal(prox.soft_thresholding_off_diagonal(array, 1), output)

    # tensor OD
    array = np.arange(27).reshape(3, 3, 3)
    output = np.array(
        [
            [[0, 0, 1], [2, 4, 4], [5, 6, 8]],
            [[9, 9, 10], [11, 13, 13], [14, 15, 17]],
            [[18, 18, 19], [20, 22, 22], [23, 24, 26]],
        ]
    )
    assert_array_equal(prox.soft_thresholding_off_diagonal(array, 1), output)

    # tensor OD, lamda is a list
    array = np.arange(27).reshape(3, 3, 3)
    output = np.array(
        [
            [[0, 0, 1], [2, 4, 4], [5, 6, 8]],
            [[9, 8, 9], [10, 13, 12], [13, 14, 17]],
            [[18, 16, 17], [18, 22, 20], [21, 22, 26]],
        ]
    )

    assert_array_equal(
        prox.soft_thresholding_off_diagonal(array, np.arange(1, 4)[:, None, None]),
        output,
    )


def test_soft_thresholding_vector():
    """Test soft_thresholding_vector function."""
    # array
    array = np.arange(3)
    output = np.array([0, 0.7763932, 1.5527864])
    assert_array_almost_equal(prox.soft_thresholding_vector(array, 0.5), output)

    # matrix
    array = np.arange(9).reshape(3, 3)
    output = np.array(
        [
            [0.0, 0.87690851, 1.79260966],
            [2.5527864, 3.50763404, 4.48152415],
            [5.10557281, 6.13835956, 7.17043864],
        ]
    )
    assert_array_almost_equal(prox.blockwise_soft_thresholding(array, 1), output)

    arr = array + array.T
    assert_array_equal(
        prox.blockwise_soft_thresholding(arr, 1),
        prox.blockwise_soft_thresholding_symmetric(arr, 1),
    )

    # tensor
    arr3 = np.array([arr] * 2)
    output = np.array(
        [
            [0.0, 3.73273876, 7.62860932],
            [3.5527864, 7.46547752, 11.44291399],
            [7.10557281, 11.19821627, 15.25721865],
        ]
    )
    out = np.array([output] * 2)
    assert_array_almost_equal(prox.blockwise_soft_thresholding(arr3, 1), out)

    assert_array_almost_equal(prox.blockwise_soft_thresholding_symmetric(arr3, 1), out)
