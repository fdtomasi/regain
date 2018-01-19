"""Graphical latent variable models selection over time via ADMM."""
from __future__ import division

import numpy as np
import warnings

from functools import partial
from six.moves import range, map, zip
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_array

from regain.admm.time_graph_lasso_ import logl
from regain.admm.time_graph_lasso_ import TimeGraphLasso
from regain.norm import l1_od_norm
from regain.prox import soft_thresholding_sign
from regain.prox import prox_logdet
from regain.prox import prox_trace_indicator
from regain.update_rules import update_rho
from regain.utils import convergence
from regain.validation import check_norm_prox


def objective(S, R, Z_0, Z_1, Z_2, W_0, W_1, W_2,
              alpha, tau, beta, eta, psi, phi):
    """Objective function for latent variable time-varying graphical lasso."""
    obj = sum(- logl(s, r) for s, r in zip(S, R))
    obj += alpha * sum(map(l1_od_norm, Z_0))
    obj += tau * sum(map(partial(np.linalg.norm, ord='nuc'), W_0))
    obj += beta * sum(map(psi, Z_2 - Z_1))
    obj += eta * sum(map(phi, W_2 - W_1))
    return obj


def latent_time_graph_lasso(
        emp_cov, alpha=1., tau=1., rho=1., beta=1., eta=1., max_iter=100,
        verbose=False, psi='laplacian', phi='laplacian', mode=None,
        tol=1e-4, rtol=1e-2, assume_centered=False,
        return_history=False, return_n_iter=True):
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
    psi, prox_psi, psi_node_penalty = check_norm_prox(psi)
    phi, prox_phi, phi_node_penalty = check_norm_prox(phi)

    Z_0 = np.zeros_like(emp_cov)
    Z_1 = np.zeros_like(Z_0)[:-1]
    Z_2 = np.zeros_like(Z_0)[1:]
    W_0 = np.zeros_like(Z_0)
    W_1 = np.zeros_like(Z_1)
    W_2 = np.zeros_like(Z_2)

    X_0 = np.zeros_like(Z_0)
    X_1 = np.zeros_like(Z_1)
    X_2 = np.zeros_like(Z_2)
    U_1 = np.zeros_like(W_1)
    U_2 = np.zeros_like(W_2)

    R_old = np.zeros_like(Z_0)
    Z_1_old = np.zeros_like(Z_1)
    Z_2_old = np.zeros_like(Z_2)
    W_1_old = np.zeros_like(W_1)
    W_2_old = np.zeros_like(W_2)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.full(emp_cov.shape[0], 3, dtype=float)
    divisor[0] -= 1
    divisor[-1] -= 1

    checks = []
    for iteration_ in range(max_iter):
        # update R
        A = Z_0 - W_0 - X_0
        A *= - rho
        A += emp_cov

        A += A.transpose(0, 2, 1)
        A /= 2.

        R = np.array([prox_logdet(a, lamda=1. / rho) for a in A])

        # update Z_0
        A = R + W_0 + X_0
        A[:-1] += Z_1 - X_1
        A[1:] += Z_2 - X_2
        A /= divisor[:, None, None]
        # soft_thresholding_ = partial(soft_thresholding, lamda=alpha / rho)
        # Z_0 = np.array(map(soft_thresholding_, A))
        A += A.transpose(0, 2, 1)
        A /= 2.

        Z_0 = soft_thresholding_sign(
            A, lamda=alpha / (rho * divisor[:, None, None]))

        # update Z_1, Z_2
        A_1 = Z_0[:-1] + X_1
        A_2 = Z_0[1:] + X_2
        if not psi_node_penalty:
            prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)
            Z_1 = .5 * (A_1 + A_2 - prox_e)
            Z_2 = .5 * (A_1 + A_2 + prox_e)
        else:
            Z_1, Z_2 = prox_psi(np.concatenate((A_1, A_2), axis=1),
                                lamda=.5 * beta / rho,
                                rho=rho, tol=tol, rtol=rtol, max_iter=max_iter)

        # update W_0
        A = Z_0 - R - X_0
        A[:-1] += W_1 - U_1
        A[1:] += W_2 - U_2
        A /= divisor[:, None, None]

        A += A.transpose(0, 2, 1)
        A /= 2.

        W_0 = np.array([prox_trace_indicator(a, lamda=tau / (rho * div))
                        for a, div in zip(A, divisor)])

        # update W_1, W_2
        A_1 = W_0[:-1] + U_1
        A_2 = W_0[1:] + U_2
        if not phi_node_penalty:
            prox_e = prox_phi(A_2 - A_1, lamda=2. * eta / rho)
            W_1 = .5 * (A_1 + A_2 - prox_e)
            W_2 = .5 * (A_1 + A_2 + prox_e)
        else:
            W_1, W_2 = prox_phi(np.concatenate((A_1, A_2), axis=1),
                                lamda=.5 * eta / rho,
                                rho=rho, tol=tol, rtol=rtol, max_iter=max_iter)

        # update residuals
        X_0 += R - Z_0 + W_0
        X_1 += Z_0[:-1] - Z_1
        X_2 += Z_0[1:] - Z_2
        U_1 += W_0[:-1] - W_1
        U_2 += W_0[1:] - W_2

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(R - Z_0 + W_0) +
            squared_norm(Z_0[:-1] - Z_1) + squared_norm(Z_0[1:] - Z_2) +
            squared_norm(W_0[:-1] - W_1) + squared_norm(W_0[1:] - W_2))

        snorm = rho * np.sqrt(
            squared_norm(R - R_old) +
            squared_norm(Z_1 - Z_1_old) + squared_norm(Z_2 - Z_2_old) +
            squared_norm(W_1 - W_1_old) + squared_norm(W_2 - W_2_old))

        check = convergence(
            obj=objective(emp_cov, R, Z_0, Z_1, Z_2, W_0, W_1, W_2,
                          alpha, tau, beta, eta, psi, phi),
            rnorm=rnorm, snorm=snorm,
            e_pri=np.sqrt(R.size + 4 * Z_1.size) * tol + rtol * max(
                np.sqrt(squared_norm(R) +
                        squared_norm(Z_1) + squared_norm(Z_2) +
                        squared_norm(W_1) + squared_norm(W_2)),
                np.sqrt(squared_norm(Z_0 - W_0) +
                        squared_norm(Z_0[:-1]) + squared_norm(Z_0[1:]) +
                        squared_norm(W_0[:-1]) + squared_norm(W_0[1:]))),
            e_dual=np.sqrt(R.size + 4 * Z_1.size) * tol + rtol * rho * (
                np.sqrt(squared_norm(X_0) +
                        squared_norm(X_1) + squared_norm(X_2) +
                        squared_norm(U_1) + squared_norm(U_2))))

        R_old = R.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()
        W_1_old = W_1.copy()
        W_2_old = W_2.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        # if iteration_ % 10 == 0 and rho > 1e-6:
        #     rho /= 2.  # see Boyd, pag 21
        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_)
        # scaled dual variables should be also rescaled
        X_0 *= rho / rho_new
        X_1 *= rho / rho_new
        X_2 *= rho / rho_new
        U_1 *= rho / rho_new
        U_2 *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [Z_0, W_0, emp_cov]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class LatentTimeGraphLasso(TimeGraphLasso):
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
        super(LatentTimeGraphLasso, self).__init__(
            alpha=alpha, beta=beta, mode=mode, rho=rho,
            tol=tol, rtol=rtol, psi=psi, max_iter=max_iter, verbose=verbose,
            bypass_transpose=bypass_transpose, assume_centered=assume_centered)
        self.tau = tau
        self.eta = eta
        self.phi = phi

    def get_observed_precision(self):
        """Getter for the observed precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.

        """
        return self.precision_ - self.latent_

    def fit(self, X, y=None):
        """Fit the GraphLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_time, n_samples, n_features), or
                (n_samples, n_features, n_time)
            Data from which to compute the covariance estimate.
            If shape is (n_samples, n_features, n_time), then set
            `bypass_transpose = False`.
        y : (ignored)
        """
        if not self.bypass_transpose:
            X = X.transpose(2, 0, 1)  # put time as first dimension
        # Covariance does not make sense for a single feature
        # X = check_array(X, allow_nd=True, estimator=self)
        # if X.ndim != 3:
        #     raise ValueError("Found array with dim %d. %s expected <= 2."
        #                      % (X.ndim, self.__class__.__name__))
        X = np.array([check_array(x, ensure_min_features=2,
                      ensure_min_samples=2, estimator=self) for x in X])

        if self.assume_centered:
            self.location_ = np.zeros((X.shape[0], 1, X.shape[2]))
        else:
            self.location_ = X.mean(1).reshape(X.shape[0], 1, X.shape[2])
        emp_cov = np.array([empirical_covariance(
            x, assume_centered=self.assume_centered) for x in X])
        self.precision_, self.latent_, self.covariance_, self.n_iter_ = \
            latent_time_graph_lasso(
                emp_cov, alpha=self.alpha, tau=self.tau, rho=self.rho,
                beta=self.beta, eta=self.eta, mode=self.mode,
                tol=self.tol, rtol=self.rtol, psi=self.psi, phi=self.phi,
                max_iter=self.max_iter, verbose=self.verbose,
                return_n_iter=True, return_history=False)
        return self
