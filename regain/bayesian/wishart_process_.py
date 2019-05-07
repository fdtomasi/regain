from __future__ import print_function

from functools import partial

import numpy as np
from scipy import linalg
from sklearn.datasets.base import Bunch
from sklearn.gaussian_process import kernels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_X_y

from regain.bayesian import stats
from regain.bayesian.gaussian_process_ import sample as sample_gp
from regain.bayesian.sampling import (GWP_construct, elliptical_slice,
                                      sample_ell, sample_hyper_kernel)
from regain.covariance.time_graphical_lasso_ import TimeGraphicalLasso


def fit(
        lp, var_prop, mu_prior, var_prior, var_Lprop, mu_Lprior, var_Lprior,
        kern, p, nu=None, t=None, n_iter=500, verbose=False, likelihood=None):
    """Sample the parameters of kernel and lower Cholesky.

    Parameters
    ----------
    lp, var_prop, mu_prior, var_prior : sampling kernel hyperparameters
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

    K = kern(t[:, None], inverse_width=lp)
    umat = sample_gp(K, nu=nu, p=p)

    L = np.tril(np.random.randn(p, p))
    # Cholesky factor for the scale matrix V = LL^{\top}

    D = GWP_construct(np.real(umat), L)
    cur_log_like = likelihood(D)

    # The struct ff is formed for the ESS procedure
    ff = Bunch(xx=umat, V=D, L=L, uut=np.array([u.dot(u.T) for u in umat.T]))

    Ltau = L[np.tril_indices_from(L)]
    L__ = np.zeros((p, p))
    L__[np.tril_indices_from(L__)] = Ltau

    samples_u = []  # np.zeros((uvec.size, niters));
    loglikes = np.zeros(n_iter)
    lps = np.zeros(n_iter)
    Ls = []

    for i in range(n_iter):
        # We first do ESS to obtain a new sample for u
        if verbose:
            print(i, "%.3e" % cur_log_like, end='\r')

        ff, cur_log_like = elliptical_slice(
            ff, umat, cur_log_like, likelihood=likelihood)

        # We now do MH for sampling the hyperparameter of the kernel
        lp, accept = sample_hyper_kernel(
            lp, var_prop, t, ff.xx, kern, mu_prior, var_prior)

        uut = ff.uut
        # We now do MH for sampling the elements in the matrix L
        Ltau = sample_ell(
            Ltau, var_Lprop, ff.xx, mu_Lprior, var_Lprior, uut=uut,
            likelihood=likelihood)

        L__[np.tril_indices_from(L__)] = Ltau
        ff['L'] = L__

        if accept:
            # new kernel parameter, recompute
            K = kern(t[:, None], inverse_width=lp)
            while True:
                try:
                    umat = sample_gp(K, nu=nu, p=p)
                    break
                except:
                    K += 1e-8 * np.eye(t.size)

            uut = np.array([u.dot(u.T) for u in umat.T])

        samples_u.append(ff.xx)
        loglikes[i] = cur_log_like
        lps[i] = lp
        Ls.append(L__)
    return samples_u, loglikes, lps, Ls


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
    A = kern(
        t_test[:, None], t_train[:, None], inverse_width=inverse_width_map)
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
    test_size = t_test.size
    # test_covariance = np.eye(test_size) - A_invKb.dot(A.T)

    return GWP_construct(u_test, L_map)


def kernel(X, Y=None, var=None, inverse_width=None, normalised=False):
    gamma = 0.5 * inverse_width
    k = var * rbf_kernel(X, Y=Y, gamma=gamma)
    if normalised:
        k *= np.sqrt(inverse_width / (2 * np.pi))
    return k


def periodic_kernel(X, Y=None, inverse_width=1):
    return kernels.ExpSineSquared(length_scale=inverse_width)(X, Y=Y)


class WishartProcess(TimeGraphicalLasso):
    def __init__(
            self, theta=100, var_prop=1, mu_prior=1, var_prior=10,
            var_Lprop=10, mu_Lprior=1, var_Lprior=1, n_iter=500, burn_in=None,
            verbose=False, assume_centered=False, kernel=None):
        self.n_iter = n_iter
        self.burn_in = n_iter // 4 if burn_in is None else burn_in
        self.verbose = verbose
        self.assume_centered = assume_centered

        # parameter initialisation
        self.theta = theta  # Inverse width
        self.var_prop = var_prop
        self.mu_prior = mu_prior
        self.var_prior = var_prior

        self.var_Lprop = var_Lprop
        self.mu_Lprior = mu_Lprior
        self.var_Lprior = var_Lprior
        self.kernel = kernel

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
        X, y = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, order="C",
            ensure_min_features=2, estimator=self)

        n_dimensions = X.shape[1]
        self.classes_, n_samples = np.unique(y, return_counts=True)
        n_times = self.classes_.size

        # n_samples = np.array([x.shape[0] for x in X])
        if self.assume_centered:
            self.location_ = np.zeros((n_times, n_dimensions))
        else:
            self.location_ = np.array(
                [X[y == cl].mean(0) for cl in self.classes_])

        # X = (X - self.location_).transpose(1, 2, 0)  # put time last
        X_center = [
            X[y == cl] - self.location_[i]
            for i, cl in enumerate(self.classes_)
        ]
        if self.kernel is None or self.kernel.lower() == 'rbf':
            kern = partial(kernel, var=1)
        else:
            kern = periodic_kernel

        self.likelihood = partial(stats.t_mvn_logpdf, X_center)

        self.nu_ = 1
        samples_u, loglikes, lps, Ls = fit(
            self.theta, self.var_prop, self.mu_prior, self.var_prior,
            self.var_Lprop, self.mu_Lprior, self.var_Lprior, kern=kern,
            t=self.classes_, nu=self.nu_, p=n_dimensions, n_iter=self.n_iter,
            verbose=self.verbose, likelihood=self.likelihood)

        # Burn in
        self.lps_after_burnin = lps[self.burn_in:]
        self.samples_u_after_burnin = samples_u[self.burn_in:]
        self.loglikes_after_burnin = loglikes[self.burn_in:]
        self.Ls_after_burnin = Ls[self.burn_in:]

        # % Select the best hyperparameters based on the loglikes_after_burnin
        pos = np.argmax(self.loglikes_after_burnin)
        self.lmap = self.lps_after_burnin[pos]
        self.Lmap = self.Ls_after_burnin[pos]
        self.u_map = self.samples_u_after_burnin[pos]
        self.D_map = GWP_construct(self.u_map, self.Lmap)

        # compatibility with sklearn
        self.covariance_ = self.D_map.T
        self.precision_ = np.array(
            [linalg.pinvh(cov) for cov in self.covariance_])
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
        X, y = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, order="C",
            ensure_min_features=2, estimator=self)

        X_center = np.array(
            [
                X[y == cl] - self.location_[i]
                for i, cl in enumerate(self.classes_)
            ])
        logp = stats.t_mvn_logpdf(X_center, self.D_map)
        return logp
