"""Solve group lasso problem via ADMM.

More information can be found in the paper linked at:
http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
"""
import numpy as np
from six.moves import range

from regain.linear_model.lasso_ import lu_factor
from regain.prox import soft_thresholding
from regain.utils import flatten


def group_lasso(A, b, lamda=1.0, groups=None, rho=1.0, alpha=1.0, max_iter=1000,
                tol=1e-4, rtol=1e-2, return_history=False):
    r"""Group Lasso solver.

    Solves the following problem via ADMM
       minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))

    The input p is a K-element vector giving the block sizes n_i, so that x_i
    is in R^{n_i}.

    Parameters
    ----------
    A : array-like, 2-dimensional
        Input matrix.
    b : array-like, 1-dimensional
        Output vector.
    lamda : float, optional
        Regularisation parameter.
    groups : list
        Groups of variables.
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

    # check valid partition
    if not np.allclose(flatten(groups), np.arange(n_features)):
        raise ValueError("Invalid partition in groups. "
                         "Groups must be non-overlapping and each variables "
                         "must be selected")

    # % save a matrix-vector multiply
    Atb = A.T.dot(b)

    # ADMM solver
    x = np.zeros(n_features)
    z = np.zeros(n_features)
    u = np.zeros(n_features)

    # % pre-factor
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
        for group in groups:
            z[group] = soft_thresholding(x_hat[group] + u[group], lamda / rho)

        # % u-update
        u += (x_hat - z)

        # % diagnostics, reporting, termination checks
        history = (
            objective(A, b, lamda, groups, x, z),  # obj

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


def objective(A, b, alpha, groups, x, z):
    # obj = 0
    # for i, group in enumerate(p):
    #     obj = obj + np.linalg.norm(z[group])
    penalty = np.sum([np.linalg.norm(z[g]) for g in groups])
    return .5 * np.sum((A.dot(x) - b) ** 2) + alpha * penalty
