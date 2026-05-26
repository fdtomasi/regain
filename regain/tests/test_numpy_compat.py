# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.
"""Smoke tests covering code paths sensitive to numpy 2.x / scipy 1.17 / sklearn 1.8.

These tests exercise public estimators end-to-end plus a few helpers that
historically broke when upstream tightened semantics: scalar-vs-1d-array
assignment (NEP rules), removal of `numpy.linalg.linalg`, NEP 50 promotion
on mixed dtypes, and the deprecated `numpy.binary_repr`.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from regain import utils
from regain.covariance.graphical_lasso_ import GraphicalLasso
from regain.covariance.latent_graphical_lasso_ import LatentGraphicalLasso
from regain.covariance.latent_time_graphical_lasso_ import LatentTimeGraphicalLasso
from regain.covariance.time_graphical_lasso_ import TimeGraphicalLasso
from regain.linear_model.group_lasso_overlap_ import GroupLassoOverlap


def _gaussian_blob(n_samples=80, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.eye(n_features) + 0.3 * rng.standard_normal((n_features, n_features))
    cov = cov @ cov.T
    return rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)


def _temporal_blob(n_samples=30, n_features=3, n_times=3, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [
        _gaussian_blob(n_samples, n_features, seed=seed + t) for t in range(n_times)
    ]
    X = np.vstack(blocks)
    y = np.repeat(np.arange(n_times), n_samples)
    return X, y


def test_graphical_lasso_fits_on_numpy2():
    X = _gaussian_blob()
    mdl = GraphicalLasso(max_iter=50).fit(X)
    assert mdl.precision_.shape == (X.shape[1], X.shape[1])
    assert np.isfinite(mdl.precision_).all()


def test_latent_graphical_lasso_fits_on_numpy2():
    X = _gaussian_blob()
    mdl = LatentGraphicalLasso(max_iter=50).fit(X)
    assert mdl.precision_.shape == (X.shape[1], X.shape[1])
    assert mdl.latent_.shape == (X.shape[1], X.shape[1])


def test_time_graphical_lasso_fits_on_numpy2():
    X, y = _temporal_blob()
    mdl = TimeGraphicalLasso(max_iter=20, assume_centered=True).fit(X, y)
    n_times = len(np.unique(y))
    assert mdl.precision_.shape == (n_times, X.shape[1], X.shape[1])


def test_latent_time_graphical_lasso_fits_on_numpy2():
    X, y = _temporal_blob()
    mdl = LatentTimeGraphicalLasso(max_iter=20, assume_centered=True).fit(X, y)
    n_times = len(np.unique(y))
    assert mdl.precision_.shape == (n_times, X.shape[1], X.shape[1])


def test_group_lasso_overlap_with_overlapping_groups():
    """Regression test for `P_star_x_bar_function`.

    Previously assigned a 1-element ndarray into a scalar slot. Numpy 2.x
    rejects this with ``ValueError: setting an array element with a sequence``.
    Overlap between groups is required to exercise the consensus path.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = X @ np.array([1.0, -0.5, 0.0, 2.0]) + 0.1 * rng.standard_normal(50)

    mdl = GroupLassoOverlap(groups=[[0, 1], [1, 2], [2, 3]], max_iter=50).fit(X, y)
    assert mdl.coef_.shape[-1] == X.shape[1]
    assert np.isfinite(mdl.coef_).all()


def test_group_lasso_overlap_singleton_groups():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3))
    y = X.sum(axis=1) + 0.1 * rng.standard_normal(40)

    mdl = GroupLassoOverlap(groups=[[0], [1], [2]], max_iter=50).fit(X, y)
    assert mdl.coef_.shape[-1] == 3


def test_mk_all_ugs_binary_repr_path():
    """Exercises `numpy.binary_repr` (deprecated in numpy 2.1)."""
    pytest.importorskip("scipy.special")
    from regain.bayesian.gwishart_inference import mk_all_ugs

    graphs = mk_all_ugs(3)
    # 3 nodes → 3 possible undirected edges → 2**3 = 8 graphs.
    assert len(graphs) == 8
    for g in graphs:
        assert g.shape == (3, 3)
        assert np.array_equal(g, g.T)  # all returned graphs are symmetric


def test_utils_upper_to_full_dtype_promotion():
    """NEP 50 changed promotion rules; verify upper_to_full handles f32 inputs."""
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    a += a.T
    upper = a[np.triu_indices(3)].astype(np.float32)
    full = utils.upper_to_full(upper)
    assert full.shape == (3, 3)
    assert np.allclose(full, a)


def test_utils_flatten_handles_ragged_list():
    """`utils.flatten` should accept ragged nested lists (numpy 2.x rejects
    these in `np.array(...)`)."""
    a = [[1, 2], [3, 4], [5]]
    out = utils.flatten(a)
    assert list(out) == [1, 2, 3, 4, 5]


def test_utils_error_norm_mixed_dtype():
    """Mixed f32/f64 input shouldn't raise under NEP 50 strict casting."""
    a = np.arange(9, dtype=np.float64).reshape(3, 3)
    a += a.T
    b = a.astype(np.float32)
    err = utils.error_norm(a, b)
    assert np.isfinite(err)
    assert_array_almost_equal(err, 0.0, decimal=4)


def test_linalg_error_import_path():
    """`numpy.linalg.linalg` was removed in numpy 2.0; verify regain uses the public path."""
    from regain.utils import LinAlgError

    # Just checking that the import resolves and matches numpy's public symbol.
    assert LinAlgError is np.linalg.LinAlgError
