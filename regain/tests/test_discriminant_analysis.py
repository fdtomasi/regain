# BSD 3-Clause License
# Copyright (c) 2019, regain authors
"""Tests for regain.discriminant_analysis.

DiscriminantAnalysis wraps a class-conditional precision estimator (e.g.
LatentTimeGraphicalLasso) and runs sklearn QDA on top. A stub estimator
keeps the test fast and isolated from the covariance solvers' convergence
behaviour.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from regain.discriminant_analysis import DiscriminantAnalysis


class _StubEstimator(BaseEstimator):
    """Minimal class-conditional Gaussian estimator.

    Produces `precision_`, `location_`, and `covariance_` per class by
    fitting an empirical Gaussian on each label group.
    """

    def fit(self, X, y):
        classes = np.unique(y)
        means = []
        precs = []
        covs = []
        for k in classes:
            Xk = X[y == k]
            mu = Xk.mean(axis=0)
            cov = np.cov(Xk, rowvar=False) + 1e-3 * np.eye(X.shape[1])
            prec = np.linalg.inv(cov)
            means.append(mu)
            precs.append(prec)
            covs.append(cov)
        self.location_ = np.array(means)
        self.precision_ = np.array(precs)
        self.covariance_ = np.array(covs)
        return self


@pytest.fixture
def two_class_blob():
    rng = np.random.default_rng(0)
    X0 = rng.standard_normal((40, 3))
    X1 = rng.standard_normal((40, 3)) + np.array([3.0, 0.0, -3.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * 40 + [1] * 40)
    return X, y


def test_discriminant_analysis_fits_and_predicts(two_class_blob):
    X, y = two_class_blob
    mdl = DiscriminantAnalysis(estimator=_StubEstimator()).fit(X, y)
    preds = mdl.predict(X)
    assert preds.shape == (X.shape[0],)
    # Two well-separated Gaussians should be classified nearly perfectly.
    assert mdl.score(X, y) > 0.9


def test_discriminant_analysis_priors_explicit(two_class_blob):
    X, y = two_class_blob
    mdl = DiscriminantAnalysis(estimator=_StubEstimator(), priors=[0.5, 0.5]).fit(X, y)
    np.testing.assert_allclose(mdl.priors_, [0.5, 0.5])


def test_discriminant_analysis_rejects_single_class():
    X = np.zeros((10, 3))
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="greater than"):
        DiscriminantAnalysis(estimator=_StubEstimator()).fit(X, y)


def test_discriminant_analysis_rejects_estimator_without_fit():
    class NoFit:
        pass

    X = np.zeros((10, 3))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="does not implement"):
        DiscriminantAnalysis(estimator=NoFit()).fit(X, y)


def test_discriminant_analysis_rejects_singleton_class():
    X = np.vstack([np.zeros((9, 3)), np.ones((1, 3))])
    y = np.array([0] * 9 + [1])
    with pytest.raises(ValueError, match="ill defined"):
        DiscriminantAnalysis(estimator=_StubEstimator()).fit(X, y)


def test_discriminant_analysis_ensure_posdef_branch(two_class_blob):
    """When `ensure_posdef=True` the wrapper rebuilds covariance from precision."""
    X, y = two_class_blob
    mdl = DiscriminantAnalysis(estimator=_StubEstimator(), ensure_posdef=True).fit(X, y)
    assert mdl.covariance_.shape == (2, 3, 3)
    assert mdl.score(X, y) > 0.9
