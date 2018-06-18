"""Test LatentTimeGraphLasso."""
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.covariance import GraphLasso as GL

from regain.covariance.graph_lasso_ import GraphLasso


def test_gl():
    """Check GraphLasso vs sklearn-graph lasso."""
    np.random.seed(2)
    X = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
    p1 = GL().fit(X).precision_
    p2 = GraphLasso().fit(X).precision_

    assert_array_almost_equal(p1, p2, 1)
