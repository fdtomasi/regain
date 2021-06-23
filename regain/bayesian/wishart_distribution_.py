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

import numpy as np
from scipy import linalg, stats
from scipy.special import multigammaln
from sklearn.base import BaseEstimator
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.utils.extmath import fast_logdet

import statsmodels.sandbox.distributions.mv_normal as mvd


class WishartDistribution(BaseEstimator):
    """Wishart distribution."""

    def __init__(self, nu, S):
        self.nu = nu
        self.S = S
        self.D = S.shape[0]
        self.inv_S = linalg.pinvh(S)

    @property
    def mean(self):
        return self.nu * self.S

    @property
    def mode(self):
        return (self.nu - self.D - 1) * self.S if self.nu > self.D + 1 else np.nan

    def log_likelihood(self, X):
        """Equivalent to scipy.

        from scipy.stats import wishart
        wishart.logpdf(X, nu, S)
        """
        nu = self.nu
        n_dim = X.shape[0]
        inv_S = self.inv_S

        logp = (nu - n_dim - 1) * fast_logdet(X)
        logp -= np.sum(X * inv_S)
        logp -= nu * n_dim * np.log(2)
        logp -= 2 * multigammaln(0.5 * nu, n_dim)
        logp -= nu * fast_logdet(self.S)
        logp /= 2.0
        return logp

    def likelihood(self, X):
        return np.exp(self.log_likelihood(X))

    def sample(self, n_samples=1):
        # return distributions.rwishart(self.nu, self.psi)
        # return distributions.rwishart(self.nu, self.inv_psi)
        return stats.wishart(df=self.nu, scale=self.S, size=n_samples)


class InverseWishartDistribution(WishartDistribution):
    """Inverse Wishart (IW) distribution."""

    def __init__(self, nu, S):
        super(InverseWishartDistribution, self).__init__(nu=nu, S=S)

    @property
    def mean(self):
        # return self.inv_S / (self.nu - self.D - 1)
        return self.S / (self.nu - self.D - 1)

    @property
    def mode(self):
        # return self.inv_S / (self.nu + self.D + 1)
        return self.S / (self.nu + self.D + 1)

    def log_likelihood(self, X):
        """Equivalent to scipy.

        from scipy.stats import invwishart
        invwishart.logpdf(X, nu, S)
        """
        nu = self.nu
        n_dim = X.shape[0]

        logp = nu * fast_logdet(self.S)
        logp -= np.sum(self.S * linalg.pinvh(X))
        logp -= (nu + n_dim + 1) * fast_logdet(X)
        logp -= nu * n_dim * np.log(2)
        logp -= 2 * multigammaln(0.5 * nu, n_dim)
        logp /= 2.0
        return logp

    # def wishartrand(self):
    #    dim = self.inv_psi.shape[0]
    #    chol = np.linalg.cholesky(self.inv_psi)
    #    foo = np.zeros((dim,dim))
    #
    #    for i in range(dim):
    #        for j in range(i+1):
    #            if i == j:
    #                foo[i,j] = np.sqrt(chi2.rvs(self.nu-(i+1)+1))
    #            else:
    #                foo[i,j]  = np.random.normal(0,1)
    #    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))
    def sample(self, n_samples=1):
        # return distributions.rwishart(self.nu, self.psi)
        # return distributions.rwishart(self.nu, self.inv_psi)
        return stats.invwishart.rvs(df=self.nu, scale=self.S, size=n_samples)


class NormalInverseWishartDistribution(InverseWishartDistribution):
    """Normal-inverse-wishart (NIW) distribution.

    Defined as Normal x InverseWishart distribution.

    References
    ----------
    Murphy's "A probabilistic approach", pag 133.
    """

    def __init__(self, mu, kappa, nu, S):
        super(NormalInverseWishartDistribution, self).__init__(nu=nu, S=S)
        self.mu = mu
        self.kappa = kappa

    def sample(self, n_samples=1):
        """A sample of the NIW is a pair (mu, Sigma).

        Sigma is a sample from the InverseWishartDistribution.
        mu is a sample from the multivariate_normal.
        """
        iw_samples = super(NormalInverseWishartDistribution, self).sample(n_samples=n_samples)
        if n_samples == 1:
            mu = np.random.multivariate_normal(self.mu, iw_samples / self.kappa)
            return (mu, iw_samples)
        return [(np.random.multivariate_normal(self.mu, Sigma / self.kappa), Sigma) for Sigma in iw_samples]

    def log_likelihood(self, mu, X):
        """Equivalent to likelihood of Normal times InverseWishart."""
        logl = super(NormalInverseWishartDistribution, self).log_likelihood(X)
        logl += stats.multivariate_normal.logpdf(mu, self.mu, X / self.kappa)

        # # equivalent to
        # nu = self.nu
        # n_dim = X.shape[0]
        #
        # logp = nu * fast_logdet(self.S)
        # logp -= np.sum(self.S * linalg.pinvh(X))
        # logp -= (nu + n_dim + 2) * fast_logdet(X)
        # logp -= nu * n_dim * np.log(2)
        # logp -= 2 * multigammaln(0.5 * nu, n_dim)
        #
        # logp -= n_dim * np.log(2 * np.pi / self.kappa)
        # logp -= self.kappa * np.linalg.multi_dot(
        #     (mu - self.mu, linalg.pinvh(X), mu - self.mu))
        # logp /= 2.

        return logl

    @property
    def mode(self):
        """Mode of the distribution (Murphy, pag 134)."""
        return (self.mu, self.S / (self.nu + self.D + 2))

    @property
    def marginals(self):
        """Marginals of the distribution (Murphy, pag 134)."""
        # mu has a multivariate student t distribution
        df = self.nu - self.D + 1
        # XXX this is univariate, need multivariate in Python
        # marginal_mu = stats.t(
        #     df=df, loc=self.mu, scale=self.S / (self.kappa * df))
        marginal_mu = mvd.MVT(df=df, mean=self.mu.ravel(), sigma=self.S / (self.kappa * df))
        marginal_sigma = InverseWishartDistribution(nu=self.nu, S=self.S)
        return (marginal_mu, marginal_sigma)

    def posterior(self, X):
        """The posterior can be shown to be NIW with updated parameters.

        References
        ----------
        Probabilistic Approach, Murphy (pag 134).
        """
        n = X.shape[0]
        nu_n = self.nu + n
        S = X.T.dot(X)

        mean = np.mean(X, axis=0)
        # S_mu = empirical_covariance(data - mean, assume_centered=True) * n

        k_0 = self.kappa
        k_n = k_0 + n

        m_0 = self.mu
        m_n = (k_0 * m_0 + n * mean) / k_n

        m_0 = np.atleast_2d(m_0)
        m_n = np.atleast_2d(m_n)

        # mm = np.atleast_2d(mean - m_0)
        # psi_n_v1 = self.psi + S_mu + k_0 * n / k_n * mm.T.dot(mm)
        S_n = self.S + S + k_0 * m_0.T.dot(m_0) - k_n * m_n.T.dot(m_n)

        return NormalInverseWishartDistribution(m_n, k_n, nu_n, S_n)

    def predictive(self):
        """Posterior predictive. See pag 135."""
        df = self.nu - self.D + 1
        return mvd.MVT(df=df, mean=self.mu.ravel(), sigma=(self.kappa + 1) * self.S / (self.kappa * df))


def main():
    n_dim = 5
    n_samples = 100
    v_0 = n_dim + 2 + 1  # v_0 > p + 2
    Sigma = make_sparse_spd_matrix(n_dim, 0.75)
    mu = np.zeros(n_dim) - 3

    x = NormalInverseWishartDistribution(mu, 1, v_0, Sigma)
    # X = x.sample(n_samples=n_samples)
    X = np.random.multivariate_normal(mu, Sigma, size=n_samples)

    # prior
    prior = NormalInverseWishartDistribution(np.zeros(n_dim), 1, v_0, np.eye(n_dim))

    z = prior.posterior(X)

    print(x)
    print(z)
    print(np.linalg.norm(z.mode[1] - Sigma))
    print(np.linalg.norm(z.marginal_mode - Sigma))
    print(np.linalg.norm(z.marginal_mean - Sigma))


if __name__ == "__main__":
    main()
