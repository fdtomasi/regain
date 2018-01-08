"""Test LatentTimeGraphLasso."""
import numpy as np
import warnings
from numpy.testing import assert_array_equal

from regain.admm.time_graph_lasso_ import TimeGraphLasso


def test_ltgl_zero():
    """Check that LatentTimeGraphLasso can handle zero data."""
    a = np.zeros((3, 3, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl = TimeGraphLasso(
            bypass_transpose=False, max_iter=1, assume_centered=True).fit(a)

    for p in mdl.precision_:
        # remove the diagonal
        p.flat[::4] = 0

    assert_array_equal(mdl.precision_, a)
    assert_array_equal(mdl.get_observed_precision(),
                       mdl.precision_)
