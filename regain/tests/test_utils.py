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
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from regain import utils


def test_suppress_stdout():
    """Test suppress_stdout function."""
    with utils.suppress_stdout():
        print("Test")


def test_ensure_filename_ending():
    """Test _ensure_filename_ending function."""
    filename = utils._ensure_filename_ending("test", ".txt")
    assert_equal(filename, "test.txt")


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
        "accuracy": 0.5555555555555556,
        "average_precision": 0.66666666666666663,
        "balanced_accuracy": 0.55,
        "dor": 1.4999999999999998,
        "f1": 0.6,
        "fall_out": 0.5,
        "false_omission_rate": 0.5,
        "fdr": 0.4,
        "fn": 2,
        "fp": 2,
        "mcc": 0.0,
        "miss_rate": 0.4,
        "nlr": 0.8,
        "npv": 0.5,
        "plr": 1.2,
        "precision": 0.6,
        "prevalence": 0.5555555555555556,
        "recall": 0.6,
        "specificity": 0.5,
        "tn": 2,
        "tp": 3,
    }
    assert_equal(utils.structure_error(a, b), result)

    b = np.eye(3) + np.eye(3, k=-1) * 1e-3
    result = {
        "accuracy": 0.7777777777777778,
        "average_precision": 0.66666666666666663,
        "balanced_accuracy": 0.8,
        "dor": 0.0,
        "f1": 0.7499999999999999,
        "fall_out": 0.0,
        "false_omission_rate": 0.3333333333333333,
        "fdr": 0.0,
        "fn": 2,
        "fp": 0,
        "miss_rate": 0.4,
        "mcc": 0.0,
        "nlr": 0.4,
        "npv": 0.6666666666666666,
        "plr": 0,
        "precision": 1.0,
        "prevalence": 0.5555555555555556,
        "recall": 0.6,
        "specificity": 1.0,
        "tn": 4,
        "tp": 3,
    }

    assert_equal(utils.structure_error(a, b, thresholding=True, eps=1e-2), result)


def test_is_pos_def_accepts_identity_and_rejects_zero():
    assert utils.is_pos_def(np.eye(3))
    assert not utils.is_pos_def(np.zeros((3, 3)))


def test_is_pos_def_eigvals_path_matches_cholesky_path():
    A = np.diag([2.0, 1.0, 0.5])
    assert utils.is_pos_def(A, chol=True)
    assert utils.is_pos_def(A, chol=False)


def test_is_pos_semidef_includes_zero_eigenvalues():
    A = np.diag([1.0, 0.0, 2.0])
    assert utils.is_pos_semidef(A)
    assert not utils.is_pos_def(A)


def test_positive_definite_handles_3d_stack():
    stack = np.array([np.eye(3), 2 * np.eye(3)])
    assert utils.positive_definite(stack)
    bad = stack.copy()
    bad[0] = np.zeros((3, 3))
    assert not utils.positive_definite(bad)


def test_ensure_posdef_makes_matrix_invertible_in_place():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    A = (A + A.T) / 2  # symmetric, generally indefinite
    utils.ensure_posdef(A)
    assert utils.is_pos_def(A)


def test_ensure_posdef_3d_stack():
    rng = np.random.default_rng(0)
    stack = np.array([rng.standard_normal((3, 3)) for _ in range(2)])
    stack = (stack + stack.transpose(0, 2, 1)) / 2
    utils.ensure_posdef(stack)
    for m in stack:
        assert utils.is_pos_def(m)


def test_ensure_posdef_inplace_false_raises():
    A = np.eye(3) - 2  # not posdef
    with pytest.raises(NotImplementedError):
        utils.ensure_posdef(A, inplace=False)


def test_threshold_zeros_values_below_threshmin():
    a = np.array([-1.0, 0.5, 1.5, 3.0])
    out = utils.threshold(a, threshmin=1.0)
    assert_array_equal(np.asarray(out), [0.0, 0.0, 1.5, 3.0])


def test_threshold_clips_values_above_threshmax():
    a = np.array([-1.0, 0.5, 1.5, 3.0])
    out = utils.threshold(a, threshmax=1.0, newval=-99.0)
    assert_array_equal(np.asarray(out), [-1.0, 0.5, -99.0, -99.0])


def test_normalize_matrix_puts_ones_on_diagonal():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 4))
    M = M @ M.T  # symmetric posdef
    utils.normalize_matrix(M)
    assert_allclose(np.diag(M), np.ones(4))


def test_compose_applies_right_to_left():
    f = utils.compose(lambda x: x + 1, lambda x: x * 2)  # f(x) = 2x + 1
    assert f(3) == 7
    assert utils.compose()(42) == 42  # empty compose is identity


def test_convert_data_to_2d_stacks_and_labels():
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((3, 2)), rng.standard_normal((5, 2))]
    X, y = utils.convert_data_to_2d(data)
    assert X.shape == (8, 2)
    assert_array_equal(y, [0, 0, 0, 1, 1, 1, 1, 1])


def test_alpha_heuristic_returns_positive_for_2d_covariance():
    rng = np.random.default_rng(0)
    cov = rng.standard_normal((10, 5))
    cov = cov.T @ cov
    alpha = utils.alpha_heuristic(cov, n_samples=100)
    assert alpha > 0


def test_alpha_heuristic_handles_3d_covariance_stack():
    rng = np.random.default_rng(0)
    cov = np.array([rng.standard_normal((10, 5)) for _ in range(3)])
    cov = np.einsum("tij,tkj->tik", cov, cov)  # batch of T outer products
    alpha = utils.alpha_heuristic(cov, n_samples=100)
    assert alpha > 0


def test_compose_chained_three_functions():
    f = utils.compose(str, lambda x: x + 10, lambda x: x * 2)
    assert f(3) == "16"  # ((3 * 2) + 10) = 16 → str


def test_save_and_load_pickle_roundtrip(tmp_path):
    obj = {"a": np.arange(5), "b": [1, 2, 3]}
    path = tmp_path / "obj"  # save_pickle appends .pkl automatically
    utils.save_pickle(obj, str(path))
    loaded = utils.load_pickle(str(path) + ".pkl")
    assert_array_equal(loaded["a"], obj["a"])
    assert loaded["b"] == obj["b"]


def test_ensure_filename_ending_idempotent():
    assert utils._ensure_filename_ending("foo.txt") == "foo.txt"
    assert utils._ensure_filename_ending("foo") == "foo.txt"
    assert utils._ensure_filename_ending("foo", [".bin", ".pkl"]) == "foo.bin"
