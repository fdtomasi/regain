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
"""Plotting utils."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import linalg


def plot_cov_2d(means, cov, sdwidth=1.0, npts=50, ax=None, c=None):
    """Plot a 2D covariance matrix.

    Parameters
    ----------
    means : list
        List of two means for the two dimensions of the covariance.
    cov : ndarray
        Covariance matrix (must be 2-dimensional).
    sdwidth : float, optional
        Width of the standard deviation.
    npts : int, optional
        Number of points used to generate the plot.
    ax : plt.ax, optional
        Pass an axis to plot the covariance there,
    c : str, optional
        Color of the plot.
    """
    tt = np.linspace(0, 2 * np.pi, 100)
    ap = np.array([np.cos(tt), np.sin(tt)])

    d, v = linalg.eigh(cov)
    d = sdwidth * np.sqrt(d)

    bp = (v.dot(np.diag(d)).dot(ap)) + means.ravel()[:, None]
    plot = plt.plot if ax is None else ax.plot
    plot(bp[0], bp[1], c=c)
    return cov[0, 1]  # return their correlation


def plot_cov_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
    """Plots an `nstd` sigma error ellipse based on the specified covariance.

    Additional keyword arguments are passed on to the ellipse patch artist.

    Parameters
    ----------
    cov : The 2x2 covariance matrix to base the ellipse on
    pos : The location of the center of the ellipse. Expects a 2-element
        sequence of [x0, y0].
    nstd : The radius of the ellipse in numbers of standard deviations.
        Defaults to 2 standard deviations.
    ax : The axis that the ellipse will be plotted on. Defaults to the
        current axis.
    Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
    A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)

    if width == 0:
        width += 0.1
    if height == 0:
        height += 0.1

    if ax is None:
        ax = plt.gca()
        plt.xlim([-(width + pos[0] + 0.1), width + pos[0] + 0.1])
        plt.ylim([-(height + pos[1] + 0.1), height + pos[1] + 0.1])
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip
