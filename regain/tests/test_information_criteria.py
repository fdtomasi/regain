# BSD 3-Clause License
# Copyright (c) 2019, regain authors
"""Tests for the BIC / EBIC information criteria in regain.scores."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from regain import scores


@pytest.fixture
def identity_pair():
    p = 4
    emp_cov = np.eye(p)
    precision = np.eye(p)
    return emp_cov, precision, p


@pytest.fixture
def temporal_identity_pair():
    p, T = 4, 3
    emp_cov = np.array([np.eye(p) for _ in range(T)])
    precision = np.array([np.eye(p) for _ in range(T)])
    return emp_cov, precision, p, T


def test_BIC_zero_off_diagonal_only_penalises_diag_difference(identity_pair):
    emp_cov, precision, p = identity_pair
    # For identity precision: ll = -p, of_nonzero = p - p = 0 → BIC = -p.
    assert_allclose(scores.BIC(emp_cov, precision), -p)


def test_EBIC_decreases_with_epsilon():
    rng = np.random.default_rng(0)
    p = 5
    emp_cov = np.eye(p)
    precision = np.eye(p) + 0.1 * rng.standard_normal((p, p))
    precision = (precision + precision.T) / 2
    s_low = scores.EBIC(emp_cov, precision, n=100, epsilon=0.1)
    s_high = scores.EBIC(emp_cov, precision, n=100, epsilon=2.0)
    # Higher epsilon → larger penalty → lower (more negative) score.
    assert s_high < s_low


def test_EBIC_m_differs_from_EBIC_when_p_large():
    p = 10
    emp_cov = np.eye(p)
    precision = np.eye(p)
    # On a diagonal precision both should equal log_likelihood since penalty=0.
    assert_allclose(scores.EBIC(emp_cov, precision), -p)
    assert_allclose(scores.EBIC_m(emp_cov, precision), -p)


def test_BIC_t_sums_over_time(temporal_identity_pair):
    emp_cov, precision, p, T = temporal_identity_pair
    # log_likelihood for each slice is -p; total ll = -p*T; nonzeros - p*T = 0.
    assert_allclose(scores.BIC_t(emp_cov, precision), -p * T)


def test_EBIC_t_temporal_consistency(temporal_identity_pair):
    emp_cov, precision, _, _ = temporal_identity_pair
    val = scores.EBIC_t(emp_cov, precision, n=50, epsilon=0.5)
    # Diagonal-only precision ⇒ penalty term vanishes ⇒ equals log_likelihood_t.
    assert_allclose(val, scores.log_likelihood_t(emp_cov, precision))


def test_EBIC_m_t_temporal_consistency(temporal_identity_pair):
    emp_cov, precision, _, _ = temporal_identity_pair
    val = scores.EBIC_m_t(emp_cov, precision, n=50, epsilon=0.5)
    assert_allclose(val, scores.log_likelihood_t(emp_cov, precision))
