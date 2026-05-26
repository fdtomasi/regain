# BSD 3-Clause License
# Copyright (c) 2019, regain authors
"""Tests for regain.validation."""

import numpy as np
import pytest
import scipy.sparse as sp

from regain import validation


def test_check_norm_prox_known_functions():
    for name in ("laplacian", "l1", "l2", "linf", "node"):
        norm, prox, is_node = validation.check_norm_prox(name)
        assert callable(norm)
        assert callable(prox)
        assert is_node == (name == "node")


def test_check_norm_prox_rejects_unknown():
    with pytest.raises(ValueError, match="not understood"):
        validation.check_norm_prox("nonsense")


def test_check_array_dimensions_promotes_2d_to_3d():
    X = np.zeros((5, 4))
    out = validation.check_array_dimensions(X, n_dimensions=3)
    assert out.shape == (1, 5, 4)


def test_check_array_dimensions_rejects_wrong_ndim():
    X = np.zeros((2, 5, 4, 3))
    with pytest.raises(ValueError, match="should have"):
        validation.check_array_dimensions(X, n_dimensions=3)


def test_check_array_dimensions_passthrough_for_list(recwarn):
    X = [np.zeros((5, 4)), np.zeros((6, 4))]
    out = validation.check_array_dimensions(X, suppress_warn_list=True)
    assert out is X


def test_check_array_dimensions_transposes_time_last():
    X = np.zeros((5, 4, 3))  # (features, features, time)
    out = validation.check_array_dimensions(X, n_dimensions=3, time_on_axis="last")
    assert out.shape == (3, 5, 4)


def test_check_input_3d_returns_metadata():
    X = np.zeros((3, 10, 4))  # 3 time points, 10 samples, 4 features
    arr, n_samples, n_dim, n_times = validation.check_input_3d(X)
    assert arr.shape == (3, 10, 4)
    assert list(n_samples) == [10, 10, 10]
    assert n_dim == 4
    assert n_times == 3


def test_check_input_3d_list_uniform_shape():
    # The current implementation requires all blocks to share the same shape.
    X = [np.zeros((10, 4)), np.zeros((10, 4)), np.zeros((10, 4))]
    arr, n_samples, n_dim, n_times = validation.check_input_3d(
        X, suppress_warn_list=True
    )
    assert n_dim == 4
    assert n_times == 3
    assert list(n_samples) == [10, 10, 10]


def test_check_input_rejects_sparse():
    with pytest.raises(TypeError, match="sparse"):
        validation.check_input(sp.csr_matrix(np.eye(4)))


def test_check_input_rejects_y_not_none():
    X = np.zeros((3, 10, 4))
    with pytest.raises(ValueError, match="y cannot be"):
        validation.check_input(X, y=np.zeros(10))
