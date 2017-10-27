"""Sparse inverse covariance selection via ADMM.

More information can be found in the paper linked at:
http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
"""
from __future__ import division

import numpy as np
from numpy import zeros
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import fast_logdet

from prox import soft_thresholding


def covsel(D, lamda=1, rho=1, alpha=1, max_iter=1000, verbose=False, tol=1e-4,
           rtol=1e-2, return_history=False):
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
    S = empirical_covariance(D)
    n = S.shape[0]

    # X = zeros(n)
    Z = zeros((n, n))
    U = zeros((n, n))

    hist = []
    count = 0
    for _ in range(max_iter):
        # x-update
        es, Q = np.linalg.eigh(rho * (Z - U) - S)
        xi = (es + np.sqrt(es ** 2 + 4 * rho)) / (2. * rho)
        X = np.dot(Q.dot(np.diag(xi)), Q.T)

        # z-update with relaxation
        Zold = Z
        X_hat = alpha * X + (1 - alpha) * Zold
        Z = soft_thresholding(X_hat + U, lamda / rho)

        U = U + (X_hat - Z)

        # diagnostics, reporting, termination checks
        history = (
            objective(S, X, Z, lamda),

            np.linalg.norm(X - Z, 'fro'),
            np.linalg.norm(-rho * (Z - Zold), 'fro'),

            np.sqrt(n) * tol + rtol * max(
                np.linalg.norm(X, 'fro'), np.linalg.norm(Z, 'fro')),
            np.sqrt(n) * tol + rtol * np.linalg.norm(rho * U, 'fro')
        )

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % history)

        hist.append(history)
        if history[1] < history[3] and history[2] < history[4]:
            if count > 10:
                break
            else:
                count += 1
        else:
            count = 0

    return X, Z, hist


def objective(S, X, Z, lamda):
    return np.sum(S * X) - fast_logdet(X) + lamda * np.linalg.norm(Z, 1)
