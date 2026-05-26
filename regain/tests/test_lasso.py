# BSD 3-Clause License
# Copyright (c) 2019, regain authors
"""Tests for the ADMM Lasso solver in regain.linear_model.lasso_."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import Lasso as SkLasso

from regain.linear_model.lasso_ import lasso, lu_factor, objective


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    n_samples, n_features = 80, 6
    A = rng.standard_normal((n_samples, n_features))
    true_w = np.array([3.0, -2.0, 0.0, 0.0, 1.5, 0.0])
    b = A @ true_w + 0.05 * rng.standard_normal(n_samples)
    return A, b, true_w


def test_lasso_recovers_sparse_signal(regression_data):
    A, b, true_w = regression_data
    x = lasso(A, b, lamda=0.1, max_iter=2000, tol=1e-6, rtol=1e-6)
    # Non-zero positions should agree with the truth (modulo small ones).
    assert np.argmax(np.abs(x)) == np.argmax(np.abs(true_w))
    # Zero positions should be (near-)zero.
    assert np.all(np.abs(x[true_w == 0]) < 0.5)


def test_lasso_matches_sklearn_objective(regression_data):
    A, b, _ = regression_data
    n = A.shape[0]
    # sklearn's Lasso minimises (1/(2n))||Ax-b||^2 + alpha*||x||_1, so to match
    # regain's (1/2)||Ax-b||^2 + lamda*||x||_1 we set alpha = lamda / n.
    lamda = 0.5
    x_admm = lasso(A, b, lamda=lamda, max_iter=5000, tol=1e-8, rtol=1e-8)
    sk = SkLasso(alpha=lamda / n, fit_intercept=False, max_iter=20000, tol=1e-10).fit(
        A, b
    )
    assert_allclose(x_admm, sk.coef_, atol=1e-2)


def test_lasso_zero_lambda_is_least_squares(regression_data):
    A, b, _ = regression_data
    x = lasso(A, b, lamda=0.0, max_iter=5000, tol=1e-9, rtol=1e-9)
    ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert_allclose(x, ls, atol=1e-3)


def test_lasso_history_is_returned_when_requested(regression_data):
    A, b, _ = regression_data
    x, hist = lasso(A, b, lamda=0.1, max_iter=10, return_history=True)
    assert x.shape == (A.shape[1],)
    assert len(hist) > 0
    # Each entry is a 5-tuple of diagnostics (obj, r_norm, s_norm, eps_pri, eps_dual).
    assert len(hist[0]) == 5


def test_lu_factor_solves_normal_equations():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((20, 5))
    rho = 1.5
    L, U = lu_factor(A, rho)
    # When n_samples >= n_features the factorisation is of (A^T A + rho I).
    M = A.T @ A + rho * np.eye(A.shape[1])
    assert_allclose(L @ U, M, atol=1e-8)


def test_lu_factor_wide_matrix_uses_woodbury():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 20))
    rho = 1.5
    L, U = lu_factor(A, rho)
    # In the wide case factorisation is of (I + (1/rho) A A^T).
    M = np.eye(A.shape[0]) + (1.0 / rho) * (A @ A.T)
    assert_allclose(L @ U, M, atol=1e-8)


def test_objective_value_consistency():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((10, 4))
    b = rng.standard_normal(10)
    x = rng.standard_normal(4)
    z = x.copy()
    val = objective(A, b, alpha=0.5, x=x, z=z)
    expected = 0.5 * np.linalg.norm(A @ x - b) ** 2 + 0.5 * np.linalg.norm(z, 1)
    assert_allclose(val, expected, atol=1e-10)
