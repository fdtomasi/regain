"""Graphical latent variable models selection over time via ADMM."""
from __future__ import division

import numpy as np
import warnings

from functools import partial
from six.moves import range
from sklearn.covariance import empirical_covariance
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.extmath import squared_norm

from regain.admm.time_graph_lasso_ import log_likelihood
from regain.norm import l1_od_norm, l1_norm
from regain.prox import soft_thresholding_od, soft_thresholding_sign
from regain.prox import blockwise_soft_thresholding, prox_linf
from regain.prox import prox_logdet, prox_laplacian
from regain.prox import prox_trace_indicator
from regain.utils import convergence


def objective(n_samples, S, R, Z_0, Z_1, Z_2, W_0, W_1, W_2,
              alpha, tau, beta, eta, psi, phi):
    """Objective function for time-varying graphical lasso."""
    obj = np.sum(-n * log_likelihood(s, r) for s, r, n in zip(S, R, n_samples))
    obj += alpha * np.sum(map(l1_od_norm, Z_0))
    obj += tau * np.sum(map(partial(np.linalg.norm, ord='nuc'), W_0))
    obj += beta * np.sum(map(psi, Z_2 - Z_1))
    obj += eta * np.sum(map(phi, W_2 - W_1))
    return obj


def time_latent_graph_lasso(
        data_list, alpha=1., tau=1., rho=1., beta=1., eta=1., max_iter=1000,
        verbose=False, psi='laplacian', phi='laplacian', assume_centered=False,
        tol=1e-4, rtol=1e-2, return_history=False, return_n_iter=True,
        mode=None):
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
    if psi == 'laplacian':
        prox_psi = prox_laplacian
        psi = squared_norm
    elif psi == 'l1':
        prox_psi = soft_thresholding_sign
        psi = l1_norm
    elif psi == 'l2':
        prox_psi = blockwise_soft_thresholding
        psi = np.linalg.norm
    elif psi == 'linf':
        prox_psi = prox_linf
        psi = partial(np.linalg.norm, ord=np.inf)
    else:
        raise ValueError("Value of `psi` not understood.")
    if phi == 'laplacian':
        prox_phi = prox_laplacian
        phi = squared_norm
    elif phi == 'l1':
        prox_phi = soft_thresholding_sign
        phi = l1_norm
    elif phi == 'l2':
        prox_phi = blockwise_soft_thresholding
        phi = np.linalg.norm
    elif phi == 'linf':
        prox_phi = prox_linf
        phi = partial(np.linalg.norm, ord=np.inf)
    else:
        raise ValueError("Value of `phi` not understood.")

    emp_cov = np.array([empirical_covariance(
        x, assume_centered=assume_centered) for x in data_list])

    n_samples = np.array([s.shape[0] for s in data_list])

    K = np.zeros_like(emp_cov)
    L = np.zeros_like(emp_cov)
    X = np.zeros_like(emp_cov)
    Z_0 = np.zeros_like(K)
    Z_1 = np.zeros_like(K)[:-1]
    Z_2 = np.zeros_like(K)[1:]
    W_0 = np.zeros_like(K)
    W_1 = np.zeros_like(K)[:-1]
    W_2 = np.zeros_like(K)[1:]
    U_0 = np.zeros_like(emp_cov)
    U_1 = np.zeros_like(emp_cov)[:-1]
    U_2 = np.zeros_like(emp_cov)[1:]
    Y_0 = np.zeros_like(emp_cov)
    Y_1 = np.zeros_like(emp_cov)[:-1]
    Y_2 = np.zeros_like(emp_cov)[1:]

    U_consensus = np.zeros_like(emp_cov)
    Y_consensus = np.zeros_like(emp_cov)
    Z_consensus = np.zeros_like(emp_cov)
    Z_consensus_old = np.zeros_like(emp_cov)
    W_consensus = np.zeros_like(emp_cov)
    W_consensus_old = np.zeros_like(emp_cov)
    R_old = np.zeros_like(emp_cov)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.zeros(emp_cov.shape[0]) + 3
    divisor[0] -= 1
    divisor[-1] -= 1
    # eta = np.divide(n_samples, divisor * rho)

    checks = []
    for iteration_ in range(max_iter):
        # update R
        A = K - L - X
        # A += np.array(map(np.transpose, A))
        # A /= 2.
        A *= - rho / n_samples[:, None, None]
        A += emp_cov
        R = np.array(map(prox_logdet, A, n_samples / rho))

        # update K, L
        K = L + R + X + Z_0 - U_0
        K[:-1] += Z_1 - U_1
        K[1:] += Z_2 - U_2
        K /= divisor[:, None, None] + 1

        L = K - R - X + W_0 - Y_0
        L[:-1] += W_1 - Y_1
        L[1:] += W_2 - Y_2
        L /= divisor[:, None, None] + 1

        # update Z_0
        # Zold = Z
        # X_hat = alpha * X + (1 - alpha) * Zold
        soft_thresholding = partial(soft_thresholding_od, lamda=alpha / rho)
        Z_0 = np.array(map(soft_thresholding, K + U_0))

        # update Z_1, Z_2
        # prox_l = partial(prox_laplacian, beta=2. * beta / rho)
        prox_e = prox_psi((K[1:] - K[:-1] + U_2 - U_1),
                          lamda=2. * beta / rho)
        Z_1 = .5 * (K[:-1] + K[1:] + U_1 + U_2 - prox_e)
        Z_2 = .5 * (K[:-1] + K[1:] + U_1 + U_2 + prox_e)

        # update W_0
        A = L + Y_0
        W_0 = np.array(map(partial(prox_trace_indicator, lamda=tau / rho), A))

        # update W_1, W_2
        prox_e = prox_phi((L[1:] - L[:-1] + Y_2 - Y_1),
                          lamda=2. * eta / rho)
        W_1 = .5 * (L[:-1] + L[1:] + Y_1 + Y_2 - prox_e)
        W_2 = .5 * (L[:-1] + L[1:] + Y_1 + Y_2 + prox_e)

        # update residuals
        X += R - K + L

        U_0 += (K - Z_0)
        U_1 += (K[:-1] - Z_1)
        U_2 += (K[1:] - Z_2)

        Y_0 += (L - W_0)
        Y_1 += (L[:-1] - W_1)
        Y_2 += (L[1:] - W_2)

        # diagnostics, reporting, termination checks
        Z_consensus = Z_0.copy()
        Z_consensus[:-1] += Z_1
        Z_consensus[1:] += Z_2
        Z_consensus /= divisor[:, None, None]

        U_consensus = U_0.copy()
        U_consensus[:-1] += U_1
        U_consensus[1:] += U_2
        U_consensus /= divisor[:, None, None]

        W_consensus = W_0.copy()
        W_consensus[:-1] += W_1
        W_consensus[1:] += W_2
        W_consensus /= divisor[:, None, None]

        Y_consensus = Y_0.copy()
        Y_consensus[:-1] += Y_1
        Y_consensus[1:] += Y_2
        Y_consensus /= divisor[:, None, None]

        check = convergence(
            obj=objective(n_samples, emp_cov, R, Z_0, Z_1, Z_2, W_0, W_1, W_2,
                          alpha, tau, beta, eta, psi, phi),
            rnorm=np.sqrt(squared_norm(K - Z_consensus) +
                          squared_norm(L - W_consensus) +
                          squared_norm(K - L - R)),
            snorm=np.sqrt(squared_norm(rho * (Z_consensus - Z_consensus_old)) +
                          squared_norm(rho * (W_consensus - W_consensus_old)) +
                          squared_norm(rho * (R - R_old))),
            e_pri=np.sqrt(np.prod(K.shape) * 2) * tol + rtol * max(
                np.sqrt(squared_norm(K) + squared_norm(L) + squared_norm(K - L)),
                np.sqrt(squared_norm(Z_consensus) + squared_norm(W_consensus) + squared_norm(R))),
            e_dual=np.sqrt(np.prod(K.shape) * 2) * tol +
                rtol * np.sqrt(squared_norm(rho * (U_consensus)) +
                               squared_norm(rho * (Y_consensus)) +
                               squared_norm(rho * (X)))
        )
        Z_consensus_old = Z_consensus.copy()
        W_consensus_old = W_consensus.copy()
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


def time_latent_graph_lasso_alternative(
        data_list, alpha=1, tau=1, rho=1, beta=1., eta=1., max_iter=1000,
        verbose=False, psi='laplacian', phi='laplacian',
        tol=1e-4, rtol=1e-2, return_history=False):
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
    if psi == 'laplacian':
        prox_psi = prox_laplacian
        psi = squared_norm
    elif psi == 'l1':
        prox_psi = soft_thresholding_sign
        psi = l1_norm
    elif psi == 'l2':
        prox_psi = blockwise_soft_thresholding
        psi = np.linalg.norm
    elif psi == 'linf':
        prox_psi = prox_linf
        psi = partial(np.linalg.norm, ord=np.inf)
    else:
        raise ValueError("Value of `psi` not understood.")
    if phi == 'laplacian':
        prox_phi = prox_laplacian
        phi = squared_norm
    elif phi == 'l1':
        prox_phi = soft_thresholding_sign
        phi = l1_norm
    elif phi == 'l2':
        prox_phi = blockwise_soft_thresholding
        phi = np.linalg.norm
    elif phi == 'linf':
        prox_phi = prox_linf
        phi = partial(np.linalg.norm, ord=np.inf)
    else:
        raise ValueError("Value of `phi` not understood.")

    S = np.array(map(empirical_covariance, data_list))
    n_samples = np.array([s.shape[0] for s in data_list])

    K = np.zeros_like(S)
    # L = np.zeros_like(S)
    # X = np.zeros_like(S)
    Z_0 = np.zeros_like(K)
    Z_1 = np.zeros_like(K)[:-1]
    Z_2 = np.zeros_like(K)[1:]
    W_0 = np.zeros_like(K)
    W_1 = np.zeros_like(K)[:-1]
    W_2 = np.zeros_like(K)[1:]
    # U_0 = np.zeros_like(S)
    # U_1 = np.zeros_like(S)[:-1]
    # U_2 = np.zeros_like(S)[1:]
    X_0 = np.zeros_like(S)
    X_1 = np.zeros_like(S)[:-1]
    X_2 = np.zeros_like(S)[1:]

    # U_consensus = np.zeros_like(S)
    # Y_consensus = np.zeros_like(S)
    Z_consensus = np.zeros_like(S)
    Z_consensus_old = np.zeros_like(S)
    W_consensus = np.zeros_like(S)
    W_consensus_old = np.zeros_like(S)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.zeros(S.shape[0]) + 3
    divisor[0] -= 1
    divisor[-1] -= 1
    # eta = np.divide(n_samples, divisor * rho)

    checks = []
    for _ in range(max_iter):
        # update R
        # A = Z_consensus - U_consensus
        A = Z_0 - W_0 - X_0
        A[:-1] += Z_1 - W_1 - X_1
        A[1:] += Z_2 - W_2 - X_2

        A += np.array(map(np.transpose, A))
        A /= 2.

        A *= - rho / n_samples[:, None, None]
        A += S

        R = np.array(map(prox_logdet, A, n_samples / (rho * divisor)))

        # update Z_0
        # Zold = Z
        # X_hat = alpha * X + (1 - alpha) * Zold
        soft_thresholding = partial(soft_thresholding_od, lamda=alpha / rho)
        Z_0 = np.array(map(soft_thresholding, R + W_0 + X_0))

        # update Z_1, Z_2
        # prox_l = partial(prox_laplacian, beta=2. * beta / rho)
        # prox_e = np.array(map(prox_l, K[1:] - K[:-1] + U_2 - U_1))
        prox_e = prox_psi(-(R[1:] - R[:-1] + W_2 - W_1 + X_2 - X_1),
                          lamda=2. * beta / rho)
        Z_1 = .5 * (R[:-1] + R[1:] + W_1 + W_2 + X_1 + X_2 - prox_e)
        Z_2 = .5 * (R[:-1] + R[1:] + W_1 + W_2 + X_1 + X_2 + prox_e)

        # update W_0
        A = Z_0 - R - X_0
        W_0 = np.array(map(partial(prox_trace_indicator, lamda=tau / rho), A))

        # update W_1, W_2
        prox_e = prox_phi(-(R[1:] - R[:-1] - Z_2 + Z_1 + X_2 - X_1),
                          lamda=2. * eta / rho)
        W_1 = .5 * (R[:-1] + R[1:] - Z_1 + Z_2 + X_1 + X_2 - prox_e)
        W_2 = .5 * (R[:-1] + R[1:] - Z_1 + Z_2 + X_1 + X_2 + prox_e)

        # update residuals
        X_0 += R - Z_0 + W_0
        X_1 += R[:-1] - Z_1 + W_1
        X_2 += R[1:] - Z_2 + W_2

        # diagnostics, reporting, termination checks
        X_consensus = X_0.copy()
        X_consensus[:-1] += X_1
        X_consensus[1:] += X_2
        X_consensus /= divisor[:, None, None]

        Z_consensus = Z_0.copy()
        Z_consensus[:-1] += Z_1
        Z_consensus[1:] += Z_2
        Z_consensus /= divisor[:, None, None]

        W_consensus = W_0.copy()
        W_consensus[:-1] += W_1
        W_consensus[1:] += W_2
        W_consensus /= divisor[:, None, None]

        check = convergence(
            obj=objective(n_samples, S, R, Z_0, Z_1, Z_2, W_0, W_1, W_2,
                          alpha, tau, beta, eta, psi, phi),
            rnorm=np.sqrt(squared_norm(R - Z_consensus + W_consensus)),
            snorm=np.sqrt(squared_norm(rho * (Z_consensus - Z_consensus_old)) +
                          squared_norm(rho * (W_consensus - W_consensus_old))),
            e_pri=np.sqrt(np.prod(K.shape) * 2) * tol + rtol * max(
                np.linalg.norm(R),
                np.sqrt(squared_norm(Z_consensus) - squared_norm(W_consensus))),
            e_dual=np.sqrt(np.prod(K.shape) * 2) * tol + rtol * np.linalg.norm(
                rho * X_consensus)
        )
        Z_consensus_old = Z_consensus.copy()
        W_consensus_old = W_consensus.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
    else:
        warnings.warn("Objective did not converge.")

    if return_history:
        return Z_consensus, W_consensus, S, checks
    return Z_consensus, W_consensus, S


class LatentTimeGraphLasso(EmpiricalCovariance):
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
