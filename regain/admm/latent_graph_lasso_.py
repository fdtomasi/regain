"""Graphical latent variable model selection via ADMM."""
from __future__ import division

import numpy as np
import warnings

from six.moves import range
from sklearn.covariance import empirical_covariance
from sklearn.utils.validation import check_array

from regain.admm.graph_lasso_ import GraphLasso, logl
from regain.norm import l1_od_norm
from regain.prox import soft_thresholding_sign
from regain.prox import prox_logdet, prox_trace_indicator
from regain.update_rules import update_rho
from regain.utils import convergence


def objective(S, R, K, L, alpha, tau):
    """Objective function for latent graphical lasso."""
    obj = - logl(S, R)
    obj += alpha * l1_od_norm(K)
    obj += tau * np.linalg.norm(L, ord='nuc')
    return obj


def latent_graph_lasso(
        emp_cov, alpha=1., tau=1., rho=1., max_iter=100,
        verbose=False, tol=1e-4, rtol=1e-2, return_history=False,
        return_n_iter=True, mode=None):
    r"""Latent variable graphical lasso solver.

    Solves the following problem via ADMM:
        min - log_likelihood(S, K-L) + alpha ||K||_{od,1} + tau ||L_i||_*

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    emp_cov : array-like
        Empirical covariance matrix.
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
    return_n_iter : bool, optional
        Return the number of iteration before convergence.
    verbose : bool, default False
        Print info at each iteration.

    Returns
    -------
    K, L : np.array, 2-dimensional, size (d x d)
        Solution to the problem.
    S : np.array, 2 dimensional
        Empirical covariance matrix.
    n_iter : int
        If return_n_iter, returns the number of iterations before convergence.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    K = np.zeros_like(emp_cov)
    L = np.zeros_like(emp_cov)
    U = np.zeros_like(emp_cov)
    R_old = np.zeros_like(emp_cov)

    checks = []
    for iteration_ in range(max_iter):
        # update R
        A = emp_cov - rho * (K - L - U)
        A += A.T
        A /= 2.
        R = prox_logdet(A, lamda=1. / rho)

        A = L + R + U
        A += A.T
        A /= 2.
        K = soft_thresholding_sign(A, lamda=alpha / rho)
        
        A = K - R - U
        A += A.T
        A /= 2.
        L = prox_trace_indicator(A, lamda=tau / rho)

        # update residuals
        U += R - K + L

        # diagnostics, reporting, termination checks
        rnorm = np.linalg.norm(R - K + L)
        snorm = rho * np.linalg.norm(R - R_old)
        check = convergence(
            obj=objective(emp_cov, R, K, L, alpha, tau),
            rnorm=rnorm, snorm=snorm,
            e_pri=np.sqrt(R.size) * tol + rtol * max(
                np.linalg.norm(R), np.linalg.norm(K - L)),
            e_dual=np.sqrt(R.size) * tol + rtol * rho * np.linalg.norm(U)
        )
        R_old = R.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_)
        # scaled dual variables should be also rescaled
        U *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [K, L, emp_cov]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class LatentGraphLasso(GraphLasso):
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

    def __init__(self, alpha=0.01, tau=1., rho=1., tol=1e-4, rtol=1e-4,
                 max_iter=100, verbose=False, assume_centered=False,
                 mode='cd'):
        super(LatentGraphLasso, self).__init__(
            alpha=alpha, rho=rho,
            tol=tol, rtol=rtol, max_iter=max_iter, verbose=verbose,
            assume_centered=assume_centered)
        self.tau = tau
        self.mode = mode

    def get_precision(self):
        """Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.
            Note that this is the observed precision matrix.

        """
        return self.precision_ - self.latent_

    def fit(self, X, y=None):
        """Fit the LatentGraphLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : (ignored)
        """
        # Covariance does not make sense for a single feature
        # X = check_array(X, ensure_min_features=2, ensure_min_samples=2,
        #                 estimator=self)
        X = check_array(X, ensure_min_features=2, ensure_min_samples=2,
                        estimator=self)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)

        emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)
        self.precision_, self.latent_, self.covariance_, self.n_iter_ = \
            latent_graph_lasso(
                emp_cov, alpha=self.alpha, tau=self.tau, rho=self.rho,
                mode=self.mode, tol=self.tol, rtol=self.rtol,
                max_iter=self.max_iter, verbose=self.verbose,
                return_n_iter=True, return_history=False)
        return self
