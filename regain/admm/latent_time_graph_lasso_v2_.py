"""Graphical latent variable models selection over time via ADMM."""
from __future__ import division

import numpy as np
import warnings

from functools import partial
from six.moves import range
from sklearn.covariance import empirical_covariance
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.extmath import squared_norm

from regain.admm.latent_time_graph_lasso_ import objective
from regain.norm import l1_od_norm, l1_norm
from regain.prox import soft_thresholding_od, soft_thresholding_sign
from regain.prox import blockwise_soft_thresholding, prox_linf
from regain.prox import prox_logdet, prox_laplacian
from regain.prox import prox_trace_indicator
from regain.utils import convergence
from regain.validation import check_norm_prox


def time_latent_graph_lasso(
        emp_cov, alpha=1., tau=1., rho=1., beta=1., eta=1., max_iter=1000,
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
    psi, prox_psi = check_norm_prox(psi)
    phi, prox_phi = check_norm_prox(phi)

    # emp_cov = np.array([empirical_covariance(
    #     x, assume_centered=assume_centered) for x in data_list])

    n_samples = np.array([s for s in [1.]])

    K = np.zeros_like(emp_cov)
    L = np.zeros_like(emp_cov)
    X = np.zeros_like(emp_cov)
    Z_0 = np.zeros_like(emp_cov)
    Z_1 = np.zeros_like(emp_cov)[:-1]
    Z_2 = np.zeros_like(emp_cov)[1:]
    W_0 = np.zeros_like(emp_cov)
    W_1 = np.zeros_like(emp_cov)[:-1]
    W_2 = np.zeros_like(emp_cov)[1:]
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
        soft_thresholding = partial(soft_thresholding_sign, lamda=alpha / rho)
        Z_0 = np.array(map(soft_thresholding, K + U_0))

        # update Z_1, Z_2
        A_1 = K[:-1] + U_1
        A_2 = K[1:] + U_2
        prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)
        Z_1 = .5 * (A_1 + A_2 - prox_e)
        Z_2 = .5 * (A_1 + A_2 + prox_e)

        # update W_0
        A = L + Y_0
        W_0 = np.array(map(partial(prox_trace_indicator, lamda=tau / rho), A))

        # update W_1, W_2
        A_1 = L[:-1] + Y_1
        A_2 = L[1:] + Y_2
        prox_e = prox_phi(A_2 - A_1, lamda=2. * eta / rho)
        W_1 = .5 * (A_1 + A_2 - prox_e)
        W_2 = .5 * (A_1 + A_2 + prox_e)

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
            e_pri=np.sqrt(np.prod(K.shape) * 3) * tol + rtol * max(
                np.sqrt(squared_norm(K) + squared_norm(L) + squared_norm(K - L)),
                np.sqrt(squared_norm(Z_consensus) + squared_norm(W_consensus) + squared_norm(R))),
            e_dual=np.sqrt(np.prod(K.shape) * 3) * tol +
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


def time_latent_graph_lasso_v2(
        emp_cov, alpha=1., tau=1., rho=1., beta=1., eta=1., max_iter=1000,
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

    # emp_cov = np.array([empirical_covariance(
    #     x, assume_centered=assume_centered) for x in data_list])

    n_samples = np.array([s for s in [1.]])

    K = np.zeros_like(emp_cov)
    L = np.zeros_like(emp_cov)
    X = np.zeros_like(emp_cov)
    Z_0 = np.zeros_like(K)
    Z_1 = np.zeros_like(K)[:-1]
    Z_2 = np.zeros_like(K)[1:]
    X_0 = np.zeros_like(K)
    X_1 = np.zeros_like(K)[:-1]
    X_2 = np.zeros_like(K)[1:]
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
        A += Z_0 - W_0 - X_0
        A[:-1] += Z_1 - W_1 - X_1
        A[1:] += Z_2 - W_2 - X_2
        A /= divisor[:, None, None] + 1

        A += np.array(map(np.transpose, A))
        A /= 2.
        A *= - rho  # / n_samples[:, None, None]
        A += emp_cov
        R = np.array(map(prox_logdet, A, n_samples / (rho)))

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
        soft_thresholding = partial(soft_thresholding_sign, lamda=alpha / rho)
        Z_0 = np.array(map(soft_thresholding, (K + U_0 + R + W_0 + X_0) / 2.))

        # update Z_1, Z_2
        A_1 = (K[:-1] + U_1 + R[:-1] + W_1 + X_1) / 2.
        A_2 = (K[1:] + U_2 + R[1:] + W_2 + X_2) / 2.
        prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)
        Z_1 = .5 * (A_2 + A_1 - prox_e)
        Z_2 = .5 * (A_2 + A_1 + prox_e)

        # update W_0
        A = (L + Y_0 - R + Z_0 - X_0) / 2.
        W_0 = np.array(map(partial(prox_trace_indicator, lamda=tau / rho), A))

        # update W_1, W_2
        A_1 = (L[:-1] + Y_1 - R[:-1] + Z_1 - X_1) / 2.
        A_2 = (L[1:] + Y_2 - R[1:] + Z_2 - X_2) / 2.
        prox_e = prox_phi(A_2 - A_1, lamda=2. * eta / rho)
        W_1 = .5 * (A_2 + A_1 - prox_e)
        W_2 = .5 * (A_2 + A_1 + prox_e)

        # update residuals
        X += R - K + L
        X_0 += (R - Z_0 + W_0)
        X_1 += (R[:-1] - Z_1 + W_1)
        X_2 += (R[1:] - Z_2 + W_2)

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

        X_consensus = X_0.copy()
        X_consensus[:-1] += X_1
        X_consensus[1:] += X_2
        X_consensus /= divisor[:, None, None]

        Y_consensus = Y_0.copy()
        Y_consensus[:-1] += Y_1
        Y_consensus[1:] += Y_2
        Y_consensus /= divisor[:, None, None]

        check = convergence(
            obj=objective(n_samples, emp_cov, R, Z_0, Z_1, Z_2, W_0, W_1, W_2,
                          alpha, tau, beta, eta, psi, phi),
            rnorm=np.sqrt(squared_norm(K - Z_consensus) +
                          squared_norm(L - W_consensus) +
                          squared_norm(Z_consensus - W_consensus - R) +
                          squared_norm(K - L - R)),
            snorm=np.sqrt(squared_norm(rho * (Z_consensus - Z_consensus_old)) +
                          squared_norm(rho * (W_consensus - W_consensus_old)) +
                          squared_norm(rho * (Z_consensus - W_consensus - (Z_consensus_old - W_consensus_old))) +
                          squared_norm(rho * (R - R_old))),
            e_pri=np.sqrt(np.prod(K.shape) * 4) * tol + rtol * max(
                np.sqrt(squared_norm(K) + squared_norm(L) + squared_norm(K - L)
                        + squared_norm(Z_consensus - W_consensus)),
                np.sqrt(squared_norm(Z_consensus) + squared_norm(W_consensus) + squared_norm(R) + squared_norm(R))),
            e_dual=np.sqrt(np.prod(K.shape) * 4) * tol +
                rtol * np.sqrt(squared_norm(rho * (U_consensus)) +
                               squared_norm(rho * (Y_consensus)) +
                               squared_norm(rho * (X_consensus)) +
                               squared_norm(rho * (X)))
        )
        Z_consensus_old = Z_consensus.copy()
        W_consensus_old = W_consensus.copy()
        R_old = R.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if (check.rnorm <= check.e_pri and check.snorm <= check.e_dual):
            break
    else:
        warnings.warn("Objective did not converge.")

    return_list = [K, L, emp_cov]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list
