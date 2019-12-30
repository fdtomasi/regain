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
from numpy.testing import assert_array_equal, assert_equal

from regain import utils


def test_suppress_stdout():
    """Test suppress_stdout function."""
    with utils.suppress_stdout():
        print("Test")


def test_ensure_filename_ending():
    """Test _ensure_filename_ending function."""
    filename = utils._ensure_filename_ending('test', '.txt')
    assert_equal(filename, 'test.txt')


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


def test_structure_error():
    """Test structure_error function."""
    a = np.eye(3) + np.eye(3, k=1)
    b = np.eye(3, k=-1) + np.eye(3)
    result = {
        'accuracy': 0.5555555555555556,
    'average_precision': 0.66666666666666663,
        'balanced_accuracy': 0.55,
        'dor': 1.4999999999999998,
        'f1': 0.6,
        'fall_out': 0.5,
        'false_omission_rate': 0.5,
        'fdr': 0.4,
        'fn': 2,
        'fp': 2,
        'mcc': 0.0,
        'miss_rate': 0.4,
        'nlr': 0.8,
        'npv': 0.5,
        'plr': 1.2,
        'precision': 0.6,
        'prevalence': 0.5555555555555556,
        'recall': 0.6,
        'specificity': 0.5,
        'tn': 2,
        'tp': 3}
    assert_equal(utils.structure_error(a, b), result)

    b = np.eye(3) + np.eye(3, k=-1) * 1e-3
    result = {
        'accuracy': 0.7777777777777778,
    'average_precision': 0.66666666666666663,
        'balanced_accuracy': 0.8,
        'dor': 0.0,
        'f1': 0.7499999999999999,
        'fall_out': 0.0,
        'false_omission_rate': 0.3333333333333333,
        'fdr': 0.0,
        'fn': 2,
        'fp': 0,
        'miss_rate': 0.4,
        'mcc': 0.0,
        'nlr': 0.4,
        'npv': 0.6666666666666666,
        'plr': 0,
        'precision': 1.0,
        'prevalence': 0.5555555555555556,
        'recall': 0.6,
        'specificity': 1.0,
        'tn': 4,
        'tp': 3}

    assert_equal(utils.structure_error(a, b, thresholding=True, eps=1e-2),
                 result)
