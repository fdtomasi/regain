""" Solve lasso problem via ADMM.

More information can be found in the paper linked at:
http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
"""
import numpy as np
from six.moves import range

from regain.prox import soft_thresholding


def lasso(A, b, lamda=1.0, rho=1.0, alpha=1.0, max_iter=1000,
          tol=1e-4, rtol=1e-2, return_history=False):
    r"""Solves the following problem via ADMM:

        minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1

    Parameters
    ----------
    A : array-like, 2-dimensional
        Input matrix.
    b : array-like, 1-dimensional
        Output vector.
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
    x : numpy.array
        Solution to the problem.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.
    """
    n_samples, n_features = A.shape

    # % save a matrix-vector multiply
    Atb = A.T.dot(b)

    # ADMM solver
    x = np.zeros(n_features)
    z = np.zeros(n_features)
    u = np.zeros(n_features)

    # % cache the factorization
    L, U = lu_factor(A, rho)

    hist = []
    for _ in range(max_iter):
        # % x-update
        q = Atb + rho * (z - u)  # % temporary value
        if n_samples >= n_features:
            x = np.linalg.lstsq(U, np.linalg.lstsq(L, q)[0])[0]
        else:
            x = q - A.T.dot(
                np.linalg.lstsq(
                    U, np.linalg.lstsq(
                        L, A.dot(q))[0])[0]) / rho
            x /= rho

        # % z-update with relaxation
        zold = z
        x_hat = alpha * x + (1 - alpha) * zold
        z = soft_thresholding(x_hat + u, lamda / rho)

        # % u-update
        u += (x_hat - z)

        # % diagnostics, reporting, termination checks
        history = (
            objective(A, b, lamda, x, z),  # obj

            np.linalg.norm(x - z),  # r norm
            np.linalg.norm(-rho * (z - zold)),  # s norm

            np.sqrt(n_features) * tol + rtol * max(
                np.linalg.norm(x), np.linalg.norm(-z)),  # eps pri
            np.sqrt(n_features) * tol + rtol * np.linalg.norm(rho * u)  # eps dual
        )

        hist.append(history)
        if history[1] < history[3] and history[2] < history[4]:
            break

    return z, history if return_history else z


def objective(A, b, alpha, x, z):
    return .5 * np.sum((A.dot(x) - b) ** 2) + alpha * np.linalg.norm(z, 1)


def lu_factor(A, rho):
    n_samples, n_features = A.shape
    if n_samples >= n_features:  # if skinny
        L = np.linalg.cholesky(A.T.dot(A) + rho * np.eye(n_features))
    else:  # if fat
        L = np.linalg.cholesky(np.eye(n_samples) + 1. / rho * A.dot(A.T))
    U = L.T
    return L, U
