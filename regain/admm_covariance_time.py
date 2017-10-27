"""Sparse inverse covariance selection via ADMM.

More information can be found in the paper linked at:
http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
"""
from __future__ import division

import numpy as np
from functools import partial
from numpy import zeros
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import fast_logdet

from regain.prox import soft_thresholding, prox_l1_od, soft_thresholding_sign
from regain.utils import convergence


def prox_laplacian(A, beta):
    """Prox for l_2 square norm, Laplacian regularisation."""
    return A / (1 + 2. * beta)


def l2_square_norm(A):
    return np.linalg.norm(A, 'fro') ** 2


def l1_od_norm(precision):
    return np.abs(precision).sum() - np.abs(np.diag(precision)).sum()


def covseltime(
        data_list, lamda=1, rho=1, beta=1,
        max_iter=1000, verbose=False,
        tol=1e-4, rtol=1e-2, return_history=False):
    """Solves the following problem via ADMM:
        minimize  trace(S*X) - log det X + lambda*||X||_1

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    D : array-like, 2-dimensional
        Input matrix.
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
    S_list = [empirical_covariance(D) for D in data_list]
    n_list = [S.shape[0] for S in S_list]
    timestamps = len(data_list)

    # X = zeros(n)
    Theta = np.array([zeros((n, n)) for n in n_list])
    Z_time_1 = np.array([zeros((n, n)) for n in n_list[:-1]])
    Z_time_0 = np.array([zeros((n, n)) for n in n_list])
    Z_time_2 = np.array([zeros((n, n)) for n in n_list[1:]])
    U_time_0 = np.array([zeros((n, n)) for n in n_list])
    U_time_1 = np.array([zeros((n, n)) for n in n_list[:-1]])
    U_time_2 = np.array([zeros((n, n)) for n in n_list[1:]])

    checks = []
    count = 0
    for _ in range(max_iter):
        # x-update
        for t in range(timestamps):
            ni = n_list[t]
            eta = ni / (3. * rho)
            if t == 0:
                A = (Z_time_0[t] + Z_time_2[t] -
                     U_time_0[t] - U_time_2[t]) / 2.
            elif t == timestamps - 1:
                A = (Z_time_0[t] + Z_time_1[t-1] -
                     U_time_0[t] - U_time_1[t-1]) / 2.
            else:
                A = (Z_time_0[t] + Z_time_1[t-1] + Z_time_2[t] -
                     U_time_0[t] - U_time_1[t-1] - U_time_2[t]) / 3.
            es, Q = np.linalg.eigh((A + A.T) / (2. * eta) - S_list[t])

            xi = es + np.sqrt(np.square(es) + 4. / eta)
            Theta[t] = (eta / 2.) * np.linalg.multi_dot((Q, np.diag(xi), Q.T))

        # z-update with relaxation
        # Zold = Z
        # X_hat = alpha * X + (1 - alpha) * Zold
        Z_time_0 = np.array(map(partial(
            soft_thresholding_sign, lamda=lamda / rho),
            Theta + U_time_0))

        # other Zs
        prox_e = np.array(map(partial(prox_laplacian, beta=2. * beta / rho),
                              Theta[1:] - Theta[:-1] + U_time_2 - U_time_1))
        Z_time_1 = .5 * (Theta[:-1] + Theta[1:] + U_time_1 + U_time_2 - prox_e)
        Z_time_2 = .5 * (Theta[:-1] + Theta[1:] + U_time_1 + U_time_2 + prox_e)

        U_time_0 += (Theta - Z_time_0)
        U_time_1 += (Theta[:-1] - Z_time_1)
        U_time_2 += (Theta[1:] - Z_time_2)

        # diagnostics, reporting, termination checks
        check = convergence(
            obj=objective(S_list, Theta, Z_time_0, Z_time_1, Z_time_2, lamda,
                          beta, l2_square_norm),
            rnorm=9,#np.linalg.norm(X - Z, 'fro'),
            snorm=9,#np.linalg.norm(-rho * (Z - Zold), 'fro'),

            e_pri=0,#np.sqrt(n) * tol + rtol * max(
                # np.linalg.norm(X, 'fro'), np.linalg.norm(Z, 'fro')),
            e_dual=0,#np.sqrt(n) * tol + rtol * np.linalg.norm(rho * U, 'fro')
        )

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            if count > 10:
                break
            else:
                count += 1
        else:
            count = 0

    return Theta, S_list, checks


def objective(S_list, Theta, Z_time_0, Z_time_1, Z_time_2, lamda, beta, psi):
    obj = np.sum(np.sum(S * X) - fast_logdet(X) for S, X in zip(S_list, Theta))
    obj += lamda * np.sum(map(l1_od_norm, Z_time_0))
    obj += beta * np.sum(psi(z2 - z1) for z1, z2 in zip(Z_time_1, Z_time_2))
    return obj
