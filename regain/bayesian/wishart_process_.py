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
from __future__ import print_function

from functools import partial

import numpy as np
from scipy import linalg, stats
from sklearn.covariance import empirical_covariance
from sklearn.datasets.base import Bunch
from sklearn.gaussian_process import kernels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_X_y
from tqdm import trange

from regain.bayesian.gaussian_process_ import sample as sample_gp
from regain.bayesian.sampling import GWP_construct, elliptical_slice, sample_ell, sample_hyper_kernel
from regain.bayesian.stats import lognstat, t_mvn_logpdf
from regain.covariance.time_graphical_lasso_ import TimeGraphicalLasso


def fit(
    theta,
    var_prop,
    prior_theta_kernel,
    var_Lprop,
    prior_ell,
    kern,
    p,
    nu=None,
    n_iter=500,
    verbose=False,
    likelihood=None,
    L=None,
):
    """Sample the parameters of kernel and lower Cholesky.

    Parameters
    ----------
    theta, var_prop, mu_prior, var_prior : sampling kernel hyperparameters
    var_Lprop, mu_Lprior, var_Lprior : sampling the elements in the matrix L
    kern : function
        Function to compute the kernel. Should return a normalised square
        matrix.
    p : int
        Dimension of the problem.
    nu : int
        Degrees of freedom, usually nu = p + 1.
    """
    if nu is None:
        nu = p + 1

    K = kern(inverse_width=theta)
    umat = sample_gp(K, nu=nu, p=p)

    if L is None:
        learn_ell = True
        L = np.tril(np.random.randn(p, p))
    else:
        learn_ell = False

    # Cholesky factor for the scale matrix V = LL^{\top}
    V = GWP_construct(np.real(umat), L)
    current_state = Bunch(xx=umat, V=V, L=L, log_likelihood=likelihood(V))

    samples_u = []  # np.zeros((uvec.size, niters));
    loglikes = np.zeros(n_iter)
    lps = np.zeros(n_iter)

    if learn_ell:
        Ltau = L[np.tril_indices_from(L)]
        L__ = np.zeros((p, p))
        L__[np.tril_indices_from(L__)] = Ltau
        Ls = []

    pbar = trange(n_iter, disable=not verbose)
    for i in pbar:
        # We first do ESS to obtain a new sample for u
        pbar.set_description("loss: {:.3e}".format(current_state.log_likelihood))

        current_state = elliptical_slice(current_state, umat, likelihood=likelihood)

        # We now do MH for sampling the hyperparameter of the kernel
        theta, accept = sample_hyper_kernel(
            theta, var_prop, np.vstack(current_state.xx), kern, prior_distr=prior_theta_kernel
        )

        if accept:
            # new kernel parameter, recompute
            K = kern(inverse_width=theta)
            while True:
                try:
                    umat = sample_gp(K, nu=nu, p=p)
                    break
                except Exception:
                    K += 1e-8 * np.eye(K.shape[0])

            current_state["xx"] = umat
            current_state["log_likelihood"] = likelihood(GWP_construct(umat, current_state["L"]))

        # We now do MH for sampling the elements in the matrix L
        # spherical normal prior, element uncorrelated
        if learn_ell:
            Ltau = sample_ell(Ltau, var_Lprop, current_state.xx, prior_distr=prior_ell, likelihood=likelihood)
            L__[np.tril_indices_from(L__)] = Ltau
            current_state["L"] = L__
            current_state["log_likelihood"] = likelihood(GWP_construct(current_state["xx"], current_state["L"]))

        samples_u.append(current_state.xx)
        loglikes[i] = current_state.log_likelihood
        lps[i] = theta
        if learn_ell:
            Ls.append(L__)
    return_list = [samples_u, loglikes, lps]
    if learn_ell:
        return_list.append(Ls)
    return return_list


def predict(t_test, t_train, u_map, L_map, kern, inverse_width_map):
    """Predict covariance matrix for t_test.

    Parameters
    ----------
    t_{test, train} : ndarray
        Test and train time points.
    u_map : type
        MAP estimate of u parameter.
    L_map : type
        MAP estimate of L parameter.
    kern : type
        Kernel function.
    inverse_width_map : float
        MAP estimate of inverse_width, the kernel parameter.

    Returns
    -------
    ndarray, shape (p, p, t_test.size)
        List of covariances at t_test.

    """
    # Compute the mean for ustar for test data
    KB = kern(t_train[:, None], inverse_width=inverse_width_map)
    A = kern(t_test[:, None], t_train[:, None], inverse_width=inverse_width_map)
    invKB = linalg.pinvh(KB)

    # u_test is the mean of the data
    A_invKb = A.dot(invKB)

    u_test = np.tensordot(A_invKb, u_map.T, axes=1).T
    # equivalent to:
    # nu, p, _ = u_map.shape
    # u_test = np.zeros((nu, p, t_test.size))
    # for i in range(nu):
    #     for j in range(p):
    #         u_test[i, j, :] = A_invKb.dot(u_map[i, j, :])

    # Covariance of test data is
    # I_p - AK^{-1}A^T
    # test_size = t_test.size
    # test_covariance = np.eye(test_size) - A_invKb.dot(A.T)

    return GWP_construct(u_test, L_map)


def _rbf_kernel(X, Y=None, var=None, inverse_width=None, normalised=False):
    gamma = 0.5 * inverse_width
    k = var * rbf_kernel(X, Y=Y, gamma=gamma)
    if normalised:
        k *= np.sqrt(inverse_width / (2 * np.pi))
    return k


def _periodic_kernel(X, Y=None, inverse_width=1):
    return kernels.ExpSineSquared(length_scale=inverse_width)(X, Y=Y)


class WishartProcess(TimeGraphicalLasso):
    def __init__(
        self,
        theta=100,
        var_prop=1,
        mu_prior=1,
        var_prior=10,
        var_Lprop=10,
        mu_Lprior=1,
        var_Lprior=1,
        n_iter=500,
        burn_in=None,
        verbose=False,
        assume_centered=False,
        kernel=None,
        learn_ell=True,
    ):
        self.n_iter = n_iter
        self.burn_in = n_iter // 4 if burn_in is None else burn_in
        self.verbose = verbose
        self.assume_centered = assume_centered

        # parameter initialisation
        self.theta = theta  # Inverse width
        self.var_prop = var_prop
        # self.mu_prior = mu_prior
        # self.var_prior = var_prior

        self.var_Lprop = var_Lprop
        # self.mu_Lprior = mu_Lprior
        # self.var_Lprior = var_Lprior
        self.kernel = kernel
        self.learn_ell = learn_ell

        mu_prior, sigma_prior = lognstat(mu_prior, var_prior)
        self.prior_theta_kernel = stats.lognorm(loc=0, s=sigma_prior, scale=np.exp(mu_prior))
        self.prior_ell = stats.norm(loc=mu_Lprior, scale=np.sqrt(var_Lprior))

    def fit(self, X, y):
        """Fit the WishartProcess model to X.

        Parameters
        ----------
        X : ndarray, shape = (n_samples * n_times, n_dimensions)
            Data matrix.
        y : ndarray, shape = (n_times,)
            Indicate the temporal belonging of each matrix.
        """
        # Covariance does not make sense for a single feature
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order="C", ensure_min_features=2, estimator=self)

        n_dimensions = X.shape[1]
        self.classes_, n_samples = np.unique(y, return_counts=True)
        n_times = self.classes_.size

        # n_samples = np.array([x.shape[0] for x in X])
        if self.assume_centered:
            self.location_ = np.zeros((n_times, n_dimensions))
        else:
            self.location_ = np.array([X[y == cl].mean(0) for cl in self.classes_])

        # X = (X - self.location_).transpose(1, 2, 0)  # put time last
        X_center = [X[y == cl] - self.location_[i] for i, cl in enumerate(self.classes_)]
        if self.kernel is None or self.kernel.lower() == "rbf":
            kern = partial(_rbf_kernel, var=1)
        else:
            kern = _periodic_kernel

        self.likelihood = partial(t_mvn_logpdf, X_center)
        self.nu_ = 1
        L = None
        if not self.learn_ell:
            cov = empirical_covariance(X)
            try:
                L = np.linalg.cholesky(cov)
            except:
                np.fill_diagonal(cov, np.sum(np.abs(cov), axis=0) + 0.01)
                L = np.linalg.cholesky(cov)

        out = fit(
            theta=self.theta,
            var_prop=self.var_prop,
            prior_theta_kernel=self.prior_theta_kernel,
            var_Lprop=self.var_Lprop,
            prior_ell=self.prior_ell,
            kern=partial(kern, self.classes_[:, None]),
            nu=self.nu_,
            p=n_dimensions,
            n_iter=self.n_iter,
            verbose=self.verbose,
            likelihood=self.likelihood,
            L=L,
        )

        if self.learn_ell:
            samples_u, loglikes, lps, Ls = out
        else:
            samples_u, loglikes, lps = out

        # Burn in
        self.lps_after_burnin = lps[self.burn_in :]
        self.samples_u_after_burnin = samples_u[self.burn_in :]
        self.loglikes_after_burnin = loglikes[self.burn_in :]

        # % Select the best hyperparameters based on the loglikes_after_burnin
        pos = np.argmax(self.loglikes_after_burnin)
        self.lmap = self.lps_after_burnin[pos]
        self.u_map = self.samples_u_after_burnin[pos]

        if self.learn_ell:
            self.Ls_after_burnin = Ls[self.burn_in :]
            self.Lmap = self.Ls_after_burnin[pos]
        else:
            self.Lmap = L

        self.D_map = GWP_construct(self.u_map, self.Lmap)

        # compatibility with sklearn
        self.covariance_ = self.D_map.T
        self.precision_ = np.array([linalg.pinvh(cov) for cov in self.covariance_])
        return self

    def score(self, X, y):
        """Computes the log-likelihood of a Gaussian data set with
        `self.covariance_` as an estimator of its covariance matrix.

        Parameters
        ----------
        X : array-like, shape = [n_samples * n_times, n_features]
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y : not used, present for API consistence purpose.

        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.

        """
        # Covariance does not make sense for a single feature
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order="C", ensure_min_features=2, estimator=self)

        X_center = np.array([X[y == cl] - self.location_[i] for i, cl in enumerate(self.classes_)])
        logp = t_mvn_logpdf(X_center, self.D_map)
        return logp
