"""Graphical latent variable models selection over time via ADMM."""
from __future__ import division

import numpy as np
import warnings

from six.moves import range
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.extmath import squared_norm

from regain.admm.time_graph_lasso_ import log_likelihood
from regain.norm import l1_od_norm, l1_norm
from regain.prox import soft_thresholding_od, soft_thresholding_sign
from regain.prox import blockwise_soft_thresholding, prox_linf
from regain.prox import prox_logdet, prox_laplacian
from regain.prox import prox_trace_indicator
from regain.utils import convergence


def objective(S, R, K, L, alpha, tau):
    """Objective function for latent graphical lasso."""
    obj = - log_likelihood(S, R)
    obj += alpha * l1_od_norm(K)
    obj += tau * np.linalg.norm(L, ord='nuc')
    return obj


def latent_graph_lasso(
        emp_cov, alpha=1., tau=1., rho=1., max_iter=1000,
        verbose=False, tol=1e-4, rtol=1e-2, return_history=False,
        return_n_iter=True, mode=None):
    r"""Time-varying latent variable graphical lasso solver.

    Solves the following problem via ADMM:
        min sum_{i=1}^T -n_i log_likelihood(K_i-L_i) + alpha ||K_i||_{od,1}
            + tau ||L_i||_*
            + beta sum_{i=2}^T Psi(K_i - K_{i-1})
            + eta sum_{i=2}^T Phi(L_i - L_{i-1})

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    data_list : list of 2-dimensional matrices.
        Input matrices.
    alpha, tau : float, optional
        Regularisation parameters.
    rho : float, optional
        Augmented Lagrangian parameter.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Absolute tolerance for convergence.
    rtol : float, optional
        Relative tolerance for convergence.
    return_history : bool, optional
        Return the history of computed values.

    Returns
    -------
    K, L : numpy.array, 3-dimensional (T x d x d)
        Solution to the problem for each time t=1...T .
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.
    """
    K = np.zeros_like(emp_cov)
    L = np.zeros_like(emp_cov)
    X = np.zeros_like(emp_cov)
    R_old = np.zeros_like(emp_cov)

    checks = []
    for iteration_ in range(max_iter):
        # update R
        A = K - L - X
        A *= - rho
        A += emp_cov
        R = prox_logdet(A, lamda=1. / rho)
        K = soft_thresholding_sign(L + R + X, lamda=alpha / rho)
        L = prox_trace_indicator(K - R - X, lamda=tau / rho)

        # update residuals
        X += R - K + L

        # diagnostics, reporting, termination checks
        check = convergence(
            obj=objective(emp_cov, R, K, L, alpha, tau),
            rnorm=np.linalg.norm(R - K + L),
            snorm=np.linalg.norm(rho * (R - R_old)),
            e_pri=np.sqrt(np.prod(R.shape)) * tol + rtol * max(
                np.linalg.norm(R), np.sqrt(squared_norm(K) - squared_norm(L))),
            e_dual=np.sqrt(np.prod(R.shape)) * tol + rtol * np.linalg.norm(
                rho * X)
        )
        R_old = R.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
    else:
        warnings.warn("Objective did not converge.")

    return_list = [K, L, emp_cov]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class LatentGraphLasso(EmpiricalCovariance):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    Parameters
    ----------
    alpha : positive float, default 0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

    mode : {'cd', 'lars'}, default 'cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : positive float, default 1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped.

    enet_tol : positive float, optional
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'.

    max_iter : integer, default 100
        The maximum number of iterations.

    verbose : boolean, default False
        If verbose is True, the objective function and dual gap are
        plotted at each iteration.

    assume_centered : boolean, default False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    graph_lasso, GraphLassoCV
    """

    def __init__(self, alpha=1., tau=1., beta=1., eta=1., mode='cd', rho=1.,
                 bypass_transpose=True, tol=1e-4, rtol=1e-4,
                 psi='laplacian', phi='laplacian', max_iter=100,
                 verbose=False, assume_centered=False):
        super(LatentTimeGraphLasso, self).__init__(assume_centered=assume_centered)
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.eta = eta
        self.rho = rho
        self.mode = mode
        self.tol = tol
        self.rtol = rtol
        self.psi = psi
        self.phi = phi
        self.max_iter = max_iter
        self.verbose = verbose
        # for splitting purposes, data may come transposed, with time in the
        # last index. Set bypass_transpose=True if X comes with time in the
        # first dimension already
        self.bypass_transpose = bypass_transpose

    def fit(self, X, y=None):
        """Fits the GraphLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : (ignored)
        """
        # Covariance does not make sense for a single feature
        # X = check_array(X, ensure_min_features=2, ensure_min_samples=2,
        #                 estimator=self)
        if not self.bypass_transpose:
            X = X.transpose(2, 0, 1)  # put time as first dimension

        if self.assume_centered:
            self.location_ = np.zeros((X.shape[0], X.shape[2]))
        else:
            self.location_ = X.mean(1)
        # emp_cov = np.array([empirical_covariance(
        #     x, assume_centered=self.assume_centered) for x in X])

        self.precision_, self.latent_, self.covariance_, self.n_iter_ = \
            time_latent_graph_lasso(
                X, alpha=self.alpha, tau=self.tau, beta=self.beta, rho=self.rho,
                eta=self.eta, mode=self.mode, tol=self.tol, rtol=self.rtol,
                max_iter=self.max_iter, verbose=self.verbose,
                return_n_iter=True, psi=self.psi, phi=self.phi,
                return_history=False, assume_centered=self.assume_centered)
        return self
