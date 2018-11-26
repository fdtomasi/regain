from __future__ import print_function

from functools import partial

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.datasets.base import Bunch
from sklearn.metrics.pairwise import rbf_kernel

from regain.bayesian import stats
from regain.bayesian.gaussian_process_ import sample as sample_gp
from regain.bayesian.sampling import elliptical_slice, sample_hyper_kernel
from regainpr.bayesian.sampling import sample_ell


def GWP_construct(umat, L, uut=None):
    """Build the sample from the GWP.

    Optimised with uut:
    uut = np.array([u.dot(u.T) for u in umat.T])
    """

    if uut is None:
        v, p, n = umat.shape
        M = np.zeros((p, p, n))
        for i in range(n):
            for j in range(v):
                Lu = L.dot(umat[j, :, i])
                LuuL = Lu[:, None].dot(Lu[None, :])
                M[..., i] += LuuL

    else:
        M = np.array(
            [np.linalg.multi_dot((L, uu_i, L.T)) for uu_i in uut]).transpose()

    # assert np.allclose(N, M)
    return M


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
    ff = Bunch(xx=umat, V=D, L=L)

    Ltau = L[np.tril_indices_from(L)]
    L__ = np.zeros((p, p))
    L__[np.tril_indices_from(L__)] = Ltau
    ff.uut = np.array([u.dot(u.T) for u in umat.T])

    samples_u = []  # np.zeros((uvec.size, niters));
    loglikes = np.zeros(n_iter)
    lps = np.zeros(n_iter)
    Ls = []

    for i in range(n_iter):
        # We first do ESS to obtain a new sample for u
        if verbose and i > 0:
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
                    umat = sample_gp(K + 1e-8 * np.eye(t.size), nu=nu, p=p)
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
    test_covariance = np.eye(test_size) - A_invKb.dot(A.T)

    return GWP_construct(u_test, L_map)


def kernel(X, Y=None, var=None, inverse_width=None, normalised=False):
    gamma = 0.5 * inverse_width
    k = var * rbf_kernel(X, Y=Y, gamma=gamma)
    if normalised:
        k *= np.sqrt(inverse_width / (2 * np.pi))
    return k


class WishartProcess(BaseEstimator):
    def __init__(
            self, theta=100, var_prop=1, mu_prior=1, var_prior=10,
            var_Lprop=10, mu_Lprior=1, var_Lprior=1, n_iter=500, burn_in=None,
            verbose=False):
        self.n_iter = n_iter
        self.burn_in = n_iter // 4 if burn_in is None else burn_in
        self.verbose = verbose

        # parameter initialisation
        self.theta = theta  # Inverse width
        self.var_prop = var_prop
        self.mu_prior = mu_prior
        self.var_prior = var_prior

        self.var_Lprop = var_Lprop
        self.mu_Lprior = mu_Lprior
        self.var_Lprior = var_Lprior

    def fit(self, X, y=None):
        """Fit.

        Parameters
        ----------
        X : ndarray, shape = (n_samples, n_dimensions, n_times)
            Data tensor.
        y : ndarray, shape = (n_times,)
            Indicate the temporal belonging of each matrix.
        """

        kern = partial(kernel, var=1)
        likelihood = partial(stats.time_multivariate_normal_logpdf, X)

        n_dims = X.shape[1]
        nu = n_dims + 1
        t = np.arange(X.shape[-1]) if y is None else y

        samples_u, loglikes, lps, Ls = fit(
            self.theta, self.var_prop, self.mu_prior, self.var_prior,
            self.var_Lprop, self.mu_Lprior, self.var_Lprior, kern=kern, t=t,
            nu=nu, p=n_dims, n_iter=self.n_iter, verbose=self.verbose,
            likelihood=likelihood)

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
