"""Test LatentTimeGraphicalLasso."""
import numpy as np
from numpy.testing import assert_array_almost_equal

try:
    # sklean >= 0.20
    from sklearn.covariance import GraphicalLasso as GL
except ImportError:
    # sklean < 0.20
    from sklearn.covariance import GraphLasso as GL

from regain.covariance.graphical_lasso_ import GraphicalLasso


def test_gl():
    """Check GraphicalLasso vs sklearn-graphical lasso."""
    np.random.seed(2)
    X = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
    p1 = GL().fit(X).precision_
    p2 = GraphicalLasso().fit(X).precision_

    assert_array_almost_equal(p1, p2, 1)
