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
"""Statistical functions."""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.utils.extmath import squared_norm


def lognormal_pdf(x, mu, sigma):
    """Lognormal pdf.

    Equivalent to
    stats.lognorm.pdf(x, loc=0, s=sigma, scale=np.exp(mu))

    Reference
    ---------
    https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma
    """
    return np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2) / (x * sigma * np.sqrt(2 * np.pi))


def lognormal_logpdf(x, mu, sigma):
    """Lognormal log pdf.

    Equivalent to
    stats.lognorm.logpdf(x, loc=0, s=sigma, scale=np.exp(mu))
    """
    return np.log(lognormal_pdf(x, mu, sigma))


def lognstat(mean, variance):
    """A lognormal distribution with mean m and variance v has mu sigma as."""
    m2 = mean * mean
    mu = np.log(m2 / np.sqrt(variance + m2))
    sigma = np.sqrt(np.log(variance / m2 + 1))
    return mu, sigma


def log_lik_frob(S, D, variance):
    """Frobenius norm log likelihood."""
    logl = -0.5 * (S.size * np.log(2.0 * np.pi * variance) + squared_norm(S - D) / variance)
    return logl


def t_mvn_logpdf(X, Cov):
    """Normal log likelihood based on Cov (mu = 0).

    Parameters
    ----------
    X : ndarray, shape = (n_times, n_samples, n_dimensions)
        Data tensor.
    Cov : ndarray, shape = (n_dimensions, n_dimensions, n_times)
        Tensor of covariance matrices over time.
    """
    logp = sum(
        x.shape[0] * multivariate_normal.logpdf(x, cov=Sigma, allow_singular=True).sum() for x, Sigma in zip(X, Cov.T)
    )
    if not isinstance(logp, float):
        logp = sum(logp)
    return logp
