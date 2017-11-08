"""Time graph lasso via forward_backward (for now only in case of l1 norm)."""
from __future__ import division

import numpy as np
import warnings

from functools import partial
from six.moves import range
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import fast_logdet, squared_norm

from regain.norm import l1_od_norm, l1_norm
from regain.prox import prox_FL
from regain.utils import convergence


def _gradient(x, S, n_samples):
    return (S - np.array(map(np.linalg.inv, x))) * n_samples[:, None, None]


def _J(x, beta, lamda, gamma, alpha, S, n_samples):
    grad_ = _gradient(x, S, n_samples)
    prox_ = prox_FL(x - gamma * grad_, beta * gamma, lamda * gamma)
    return x + alpha * (prox_ - x)


def choose_alpha(alpha, x, S, n_samples, beta, lamda, gamma, theta=.99, max_iter=1000):
    """Choose alpha for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741
    """
    eps = .5
    partial_J = partial(_J, x, beta=beta, lamda=lamda,
                        gamma=gamma, S=S, n_samples=n_samples)
    partial_f = partial(_f, n_samples=n_samples, S=S)
    gradient_ = _gradient(x, S, n_samples)
    for i in range(max_iter):
        iter_diff = partial_J(alpha=alpha) - x
        obj_diff = partial_f(K=partial_J(alpha=alpha)) - partial_f(K=x)
        if obj_diff - _scalar_product_3d(iter_diff, gradient_) <= theta / (gamma * alpha) * squared_norm(iter_diff) + 1e-16:
            return alpha

        alpha *= eps
    return alpha


def _scalar_product_3d(x, y):
    return np.sum([xx.T.dot(yy) for xx, yy in zip(x, y)])


def log_likelihood(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def _f(n_samples, S, K):
    return np.sum(-n * log_likelihood(emp_cov, precision)
                  for emp_cov, precision, n in zip(S, K, n_samples))


def objective(n_samples, S, K, lamda, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = np.sum(-n * log_likelihood(emp_cov, precision)
                 for emp_cov, precision, n in zip(S, K, n_samples))
    obj += lamda * np.sum(map(l1_od_norm, K))
    obj += beta * np.sum(map(psi, K[1:] - K[:-1]))
    return obj


def time_graph_lasso(
        data_list, lamda=1, beta=1,
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

    K = np.zeros_like(S)
    # divisor for consensus variables, accounting for two less matrices
    divisor = np.zeros(S.shape[0]) + 3
    divisor[0] -= 1
    divisor[-1] -= 1

    checks = []
    alpha = 1
    Kold = np.ones_like(S) + 5000
    for _ in range(max_iter):
        for k in range(S.shape[0]):
            K[k].flat[::K.shape[1] + 1] = 1
        alpha_old = alpha

        # choose a gamma
        gamma = .75

        # total variation
        Y = _J(K, beta, lamda, gamma, 1, S, n_samples)
        alpha = choose_alpha(alpha_old, K, S, n_samples, beta, lamda, gamma)
        alpha = 1
        K = Kold + alpha * (Y - Kold)

        check = convergence(
            obj=objective(n_samples, S, K, lamda, beta, l1_norm),
            rnorm=np.linalg.norm(K - Kold),
            snorm=20, # np.linalg.norm(
                # rho * (Z_consensus - Z_consensus_old)),

            e_pri=30, # np.sqrt(np.prod(K.shape)) * tol + rtol * max(
                # np.linalg.norm(K), np.linalg.norm(Z_consensus)),
            e_dual=30 # np.sqrt(np.prod(K.shape)) * tol + rtol * np.linalg.norm(
                # rho * U_consensus)
        )

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        # if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
        #     break
        if check.rnorm <= tol:
            break
        Kold = K.copy()
    else:
        warnings.warn("Objective did not converge.")

    if return_history:
        return K, S, checks
    return K, S
