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
from regain.prox import prox_logdet
from regain.prox import soft_thresholding_od, soft_thresholding_sign
from regain.utils import convergence
from regain.validation import check_norm_prox


def log_likelihood(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def log_likelihood_trace(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    # return fast_logdet(precision) - np.trace(emp_cov * precision)
    return np.trace(emp_cov.dot(precision))


def objective(S, K, Z_0, Z_1, Z_2, alpha, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = np.sum(- log_likelihood(emp_cov, precision)
                 for emp_cov, precision in zip(S, K))
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

    K = np.zeros_like(emp_cov)
    Z_0 = np.zeros_like(emp_cov)
    Z_1 = np.zeros_like(emp_cov)[:-1]
    Z_2 = np.zeros_like(emp_cov)[1:]
    U_0 = np.zeros_like(emp_cov)
    U_1 = np.zeros_like(emp_cov)[:-1]
    U_2 = np.zeros_like(emp_cov)[1:]

    Z_0_old = np.zeros_like(Z_0)
    Z_1_old = np.zeros_like(Z_1)
    Z_2_old = np.zeros_like(Z_2)
    U_consensus = np.zeros_like(emp_cov)
    Z_consensus = np.zeros_like(emp_cov)
    Z_consensus_old = np.zeros_like(emp_cov)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.full(emp_cov.shape[0], 3, dtype=float)
    divisor[0] -= 1
    divisor[-1] -= 1

    checks = []
    for iteration_ in range(max_iter):
        # x-update
        A = K + U_0

        A *= - rho
        A += emp_cov

        Z_0 = np.array([prox_logdet(a, lamda=1. / rho) for a in A])

        # z-update with relaxation
        A = Z_0 - U_0
        A[:-1] += Z_1 - U_1
        A[1:] += Z_2 - U_2
        A /= divisor[:, None, None]
        soft_thresholding = partial(soft_thresholding_sign, lamda=alpha / rho)
        K = np.array(map(soft_thresholding, A))

        # other Zs
        A_1 = K[:-1] + U_1
        A_2 = K[1:] + U_2
        prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)
        Z_1 = .5 * (A_1 + A_2 - prox_e)
        Z_2 = .5 * (A_1 + A_2 + prox_e)

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
            obj=objective(emp_cov, Z_0, K, Z_1, Z_2, alpha, beta, psi),
            rnorm=np.sqrt(squared_norm(K - Z_0) +
                          squared_norm(K[:-1] - Z_1) +
                          squared_norm(K[1:] - Z_2)),
            snorm=np.sqrt(squared_norm(rho * (Z_0 - Z_0_old)) +
                          squared_norm(rho * (Z_1 - Z_1_old)) +
                          squared_norm(rho * (Z_2 - Z_2_old))),
            e_pri=np.sqrt(np.prod(K.shape[1:]) * (3 * K.shape[0] - 2)) * tol + rtol * max(
                np.sqrt(squared_norm(Z_0) + squared_norm(Z_1) + squared_norm(Z_2)), np.sqrt(squared_norm(K) + squared_norm(K[:-1]) + squared_norm(K[1:]))),
            e_dual=np.sqrt(np.prod(K.shape[1:]) * (3 * K.shape[0] - 2)) * tol + rtol * np.sqrt(
                squared_norm(rho * U_0) + squared_norm(rho * U_1) +
                squared_norm(rho * U_2))
        )
        # check = convergence(
        #     obj=objective(emp_cov, Z_0, K, Z_1, Z_2, alpha, beta, psi),
        #     rnorm=np.linalg.norm(K - Z_consensus),
        #     snorm=np.linalg.norm(rho * (Z_consensus - Z_consensus_old)),
        #     e_pri=np.sqrt(np.prod(K.shape)) * tol + rtol * max(
        #         np.linalg.norm(K), np.linalg.norm(Z_consensus)),
        #     e_dual=np.sqrt(np.prod(K.shape)) * tol + rtol * np.linalg.norm(
        #         rho * U_consensus)
        # )
        Z_consensus_old = Z_consensus.copy()
        Z_0_old = Z_0.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()

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
