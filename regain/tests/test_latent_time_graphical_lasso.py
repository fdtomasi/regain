"""Test LatentTimeGraphicalLasso."""
import numpy as np
import warnings

from numpy.testing import assert_array_equal

from regain.covariance.latent_time_graphical_lasso_ import LatentTimeGraphicalLasso


def test_ltgl_zero():
    """Check that LatentTimeGraphicalLasso can handle zero data."""
    x = np.zeros((9, 3))
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl = LatentTimeGraphicalLasso(
            max_iter=1, assume_centered=True).fit(x, y)

    for p in mdl.precision_:
        # remove the diagonal
        p.flat[::4] = 0

    assert_array_equal(mdl.precision_, np.zeros((3, 3, 3)))
    assert_array_equal(mdl.latent_, np.zeros((3, 3, 3)))
    assert_array_equal(
        mdl.get_observed_precision(), mdl.precision_ - mdl.latent_)


def test_ltgl_prox_l1():
    """Check that LatentTimeGraphicalLasso can handle zero data."""
    x = np.zeros((9, 3))
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl = LatentTimeGraphicalLasso(
            psi='l2', phi='node', max_iter=1, assume_centered=True).fit(x, y)

    for p in mdl.precision_:
        # remove the diagonal
        p.flat[::4] = 0

    assert_array_equal(mdl.precision_, np.zeros((3, 3, 3)))
    assert_array_equal(mdl.latent_, np.zeros((3, 3, 3)))
    assert_array_equal(
        mdl.get_observed_precision(), mdl.precision_ - mdl.latent_)
