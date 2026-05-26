# BSD 3-Clause License
# Copyright (c) 2019, regain authors
"""Tests for the ADMM Group Lasso solver in regain.linear_model.group_lasso_."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from regain.linear_model.group_lasso_ import group_lasso, objective


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    n_samples, n_features = 100, 8
    A = rng.standard_normal((n_samples, n_features))
    # First group of 4 is active, second group is zero.
    true_w = np.array([2.0, -1.5, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0])
    b = A @ true_w + 0.05 * rng.standard_normal(n_samples)
    return A, b, true_w


def test_group_lasso_zeroes_inactive_group(regression_data):
    A, b, true_w = regression_data
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    x = group_lasso(A, b, lamda=1.0, groups=groups, max_iter=2000, tol=1e-6, rtol=1e-6)
    # The inactive group should be much smaller than the active one.
    active = np.linalg.norm(x[:4])
    inactive = np.linalg.norm(x[4:])
    assert inactive < 0.1 * active


def test_group_lasso_zero_lambda_approaches_least_squares(regression_data):
    """At lambda=0 the ADMM iterate should converge near the LS solution
    (within ADMM tolerance; not bit-exact)."""
    A, b, _ = regression_data
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    x = group_lasso(
        A, b, lamda=0.0, groups=groups, max_iter=10000, tol=1e-12, rtol=1e-12
    )
    ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert_allclose(x, ls, atol=5e-2)


def test_group_lasso_rejects_invalid_partition():
    A = np.zeros((5, 4))
    b = np.zeros(5)
    # Groups don't form a valid partition of {0..n_features-1}.
    with pytest.raises(ValueError):
        group_lasso(A, b, lamda=0.1, groups=[[0, 1], [2, 3, 3]])


def test_group_lasso_returns_history_when_requested(regression_data):
    A, b, _ = regression_data
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    x, hist = group_lasso(
        A, b, lamda=0.5, groups=groups, max_iter=20, return_history=True
    )
    assert x.shape == (A.shape[1],)
    assert len(hist) > 0
    assert len(hist[0]) == 5


def test_group_lasso_objective_decomposes_correctly():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((20, 6))
    b = rng.standard_normal(20)
    groups = [[0, 1, 2], [3, 4, 5]]
    x = z = rng.standard_normal(6)
    val = objective(A, b, alpha=0.7, groups=groups, x=x, z=z)
    expected_data = 0.5 * np.linalg.norm(A @ x - b) ** 2
    expected_pen = 0.7 * sum(np.linalg.norm(z[g]) for g in groups)
    assert_allclose(val, expected_data + expected_pen, atol=1e-10)
