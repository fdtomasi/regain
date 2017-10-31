"""Sparse inverse covariance selection via ADMM.

More information can be found in the paper linked at:
http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
"""
from __future__ import division

import numpy as np
import warnings

from functools import partial
from six.moves import range
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import fast_logdet

from regain.norm import l2_square_norm, l1_od_norm
from regain.prox import prox_logdet, prox_laplacian, soft_thresholding_sign
from regain.utils import convergence


def covseltime(
        data_list, lamda=1, rho=1, beta=1,
        max_iter=1000, verbose=False,
        tol=1e-4, rtol=1e-2, return_history=False):
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
    S = np.array(map(empirical_covariance, data_list))
    n_samples = np.array([s.shape[0] for s in data_list])

    Theta = np.zeros_like(S)
    # Z_0 = np.zeros_like(Theta)
    # Z_1 = np.zeros_like(Theta)[:-1]
    # Z_2 = np.zeros_like(Theta)[1:]
    U_0 = np.zeros_like(S)
    U_1 = np.zeros_like(S)[:-1]
    U_2 = np.zeros_like(S)[1:]

    U_consensus = np.zeros_like(S)
    Z_consensus = np.zeros_like(S)
    Z_consensus_old = np.zeros_like(S)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.zeros(S.shape[0]) + 3
    divisor[0] = divisor[-1] = 2
    eta = np.divide(n_samples, divisor * rho)

    checks = []
    for _ in range(max_iter):
        # x-update
        A = Z_consensus - U_consensus
        A += np.array(map(np.transpose, A))
        A /= 2.
        A /= eta[:, np.newaxis, np.newaxis]
        A -= S
        Theta = np.array(map(prox_logdet, A, eta))

        # z-update with relaxation
        # Zold = Z
        # X_hat = alpha * X + (1 - alpha) * Zold
        soft_thresholding = partial(soft_thresholding_sign, lamda=lamda / rho)
        Z_0 = np.array(map(soft_thresholding, Theta + U_0))

        # other Zs
        # prox_l = partial(prox_laplacian, beta=2. * beta / rho)
        # prox_e = np.array(map(prox_l, Theta[1:] - Theta[:-1] + U_2 - U_1))
        prox_e = prox_laplacian(Theta[1:] - Theta[:-1] + U_2 - U_1,
                                beta=2. * beta / rho)
        Z_1 = .5 * (Theta[:-1] + Theta[1:] + U_1 + U_2 - prox_e)
        Z_2 = .5 * (Theta[:-1] + Theta[1:] + U_1 + U_2 + prox_e)

        U_0 += (Theta - Z_0)
        U_1 += (Theta[:-1] - Z_1)
        U_2 += (Theta[1:] - Z_2)

        # diagnostics, reporting, termination checks
        Z_consensus = Z_0.copy()
        Z_consensus[:-1] += Z_1
        Z_consensus[1:] += Z_2
        U_consensus = U_0.copy()
        U_consensus[:-1] += U_1
        U_consensus[1:] += U_2

        np.divide(Z_consensus, divisor[:, np.newaxis, np.newaxis], out=Z_consensus)
        np.divide(U_consensus, divisor[:, np.newaxis, np.newaxis], out=U_consensus)

        check = convergence(
            obj=objective(S, Theta, Z_0, Z_1, Z_2, lamda,
                          beta, l2_square_norm),
            rnorm=np.linalg.norm(Theta - Z_consensus),
            snorm=np.linalg.norm(
                rho * (Z_consensus - Z_consensus_old)),

            e_pri=np.sqrt(np.prod(Theta.shape)) * tol + rtol * max(
                np.linalg.norm(Theta), np.linalg.norm(Z_consensus)),
            e_dual=np.sqrt(np.prod(Theta.shape)) * tol + rtol * np.linalg.norm(
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

    if return_history:
        return Theta, S, checks
    return Theta, S


def objective(S, Theta, Z_0, Z_1, Z_2, lamda, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = np.sum(np.sum(emp_cov * precision) - fast_logdet(precision)
                 for emp_cov, precision in zip(S, Theta))
    obj += lamda * np.sum(map(l1_od_norm, Z_0))
    obj += beta * np.sum(psi(z2 - z1) for z1, z2 in zip(Z_1, Z_2))
    return obj
