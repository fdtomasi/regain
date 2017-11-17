"""Sparse inverse covariance selection over time via ADMM.

More information can be found in the paper linked at:
[cite Hallac]
"""
from __future__ import division

import numpy as np
import warnings

from functools import partial
from six.moves import range
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import fast_logdet, squared_norm

from regain.norm import l1_od_norm, l1_norm
from regain.prox import prox_logdet, prox_laplacian
from regain.prox import soft_thresholding_od, soft_thresholding_sign
from regain.prox import blockwise_soft_thresholding, prox_linf
from regain.utils import convergence
from regain.validation import check_norm_prox


def log_likelihood(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def objective(S, K, Z_0, Z_1, Z_2, alpha, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = np.sum(- log_likelihood(emp_cov, precision)
                 for emp_cov, precision, n in zip(S, K))
    obj += alpha * np.sum(map(l1_od_norm, Z_0))
    obj += beta * np.sum(map(psi, Z_2 - Z_1))
    return obj


def time_graph_lasso(
        emp_cov, alpha=1, rho=1, beta=1, max_iter=1000,
        verbose=False, psi='laplacian',
        tol=1e-4, rtol=1e-2, return_history=False, return_n_iter=True):
    """Time-varying graphical lasso solver.

    Solves the following problem via ADMM:
        minimize  trace(S*X) - log det X + lambda*||X||_1

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    data_list : list of 2-dimensional matrices.
        Input matrices.
    lamda : float, optional
        Regularisation parameter.
    rho : float, optional
        Augmented Lagrangian parameter.
    alpha : float, optional
        Over-relaxation parameter (typically between 1.0 and 1.8).
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
    X : numpy.array, 2-dimensional
        Solution to the problem.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.
    """
    psi, prox_psi = check_norm_prox(psi)

    # S = np.array(map(empirical_covariance, data_list))
    # n_samples = np.array([1. for data in data_list])

    # K = np.zeros_like(emp_cov)
    Z_0 = np.zeros_like(emp_cov)
    Z_1 = np.zeros_like(emp_cov)[:-1]
    Z_2 = np.zeros_like(emp_cov)[1:]
    U_0 = np.zeros_like(emp_cov)
    U_1 = np.zeros_like(emp_cov)[:-1]
    U_2 = np.zeros_like(emp_cov)[1:]

    U_consensus = np.zeros_like(emp_cov)
    Z_consensus = np.zeros_like(emp_cov)
    Z_consensus_old = np.zeros_like(emp_cov)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.zeros(emp_cov.shape[0]) + 3
    divisor[0] -= 1
    divisor[-1] -= 1

    checks = []
    for iteration_ in range(max_iter):
        # x-update
        # A = Z_consensus - U_consensus
        A = Z_0 - U_0
        A[:-1] += Z_1 - U_1
        A[1:] += Z_2 - U_2
        A /= divisor[:, None, None]

        # A += np.array(map(np.transpose, A))
        # A /= 2.

        # A *= - rho / n_samples[:, None, None]
        A *= - rho
        A += emp_cov

        K = np.array([prox_logdet(a, lamda=1. / rho) for a in A])

        # z-update with relaxation
        # Zold = Z
        # X_hat = alpha * X + (1 - alpha) * Zold
        soft_thresholding = partial(soft_thresholding_sign, lamda=alpha / rho)
        Z_0 = np.array(map(soft_thresholding, K + U_0))

        # other Zs
        if beta != 0:
            A_1 = K[:-1] + U_1
            A_2 = K[1:] + U_2
            prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)
            Z_1 = .5 * (A_1 + A_2 - prox_e)
            Z_2 = .5 * (A_1 + A_2 + prox_e)
        else:
            Z_1 = Z_0[:-1].copy()
            Z_2 = Z_0[1:].copy()

        # update residuals
        U_0 += K - Z_0
        U_1 += K[:-1] - Z_1
        U_2 += K[1:] - Z_2

        # diagnostics, reporting, termination checks
        Z_consensus = Z_0.copy()
        Z_consensus[:-1] += Z_1
        Z_consensus[1:] += Z_2
        Z_consensus /= divisor[:, None, None]

        U_consensus = U_0.copy()
        U_consensus[:-1] += U_1
        U_consensus[1:] += U_2
        U_consensus /= divisor[:, None, None]

        check = convergence(
            obj=objective(emp_cov, K, Z_0, Z_1, Z_2, alpha, beta, psi),
            rnorm=np.linalg.norm(K - Z_consensus),
            snorm=np.linalg.norm(rho * (Z_consensus - Z_consensus_old)),
            e_pri=np.sqrt(np.prod(K.shape)) * tol + rtol * max(
                np.linalg.norm(K), np.linalg.norm(Z_consensus)),
            e_dual=np.sqrt(np.prod(K.shape)) * tol + rtol * np.linalg.norm(
                rho * U_consensus)
        )
        Z_consensus_old = Z_consensus.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
    else:
        warnings.warn("Objective did not converge.")

    return_list = [K, emp_cov]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list
