# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Test LatentTimeGraphicalLasso."""
import warnings

import numpy as np
from numpy.testing import assert_array_equal

from regain.covariance.time_graphical_lasso_ import TimeGraphicalLasso
from regain.forward_backward.time_graphical_lasso_ import (
    TimeGraphicalLassoForwardBackward,
)


def test_tgl_zero():
    """Check that TimeGraphicalLasso can handle zero data."""
    x = np.zeros((9, 3))
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl = TimeGraphicalLasso(max_iter=1, assume_centered=True).fit(x, y)

    for p in mdl.precision_:
        # remove the diagonal
        p.flat[::4] = 0

    assert_array_equal(mdl.precision_, np.zeros((3, 3, 3)))
    assert_array_equal(mdl.get_observed_precision(), mdl.precision_)


def test_tgl_fb_zero():
    """Check that TimeGraphicalLasso forwarb-backward can handle zero data."""
    x = np.zeros((9, 3))
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl = TimeGraphicalLassoForwardBackward(max_iter=1, assume_centered=True).fit(
            x, y
        )

    for p in mdl.precision_:
        # remove the diagonal
        p.flat[::4] = 0

    assert_array_equal(mdl.precision_, np.zeros((3, 3, 3)))
    assert_array_equal(mdl.get_observed_precision(), mdl.precision_)
