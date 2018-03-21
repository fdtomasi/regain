"""Time graph lasso via forward_backward (for now only in case of l1 norm)."""
from __future__ import division

import warnings
from functools import partial

import numpy as np
from six.moves import map, range, zip
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import squared_norm

from regain.covariance.time_graph_lasso_ import loss
from regain.norm import l1_norm, l1_od_norm
from regain.prox import prox_FL
from regain.update_rules import update_gamma
from regain.utils import convergence


def penalty(precision, alpha, beta, psi):
    obj = alpha * sum(map(l1_od_norm, precision))
    obj += beta * sum(map(psi, precision[1:] - precision[:-1]))
    return obj


def objective(n_samples, emp_cov, precision, alpha, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = loss(emp_cov, precision, n_samples=n_samples)
    obj += penalty(precision, alpha, beta, psi)
    return obj


def grad_loss(x, emp_cov, n_samples):
    """Gradient of the loss function for the time-varying graphical lasso."""
    grad = emp_cov - np.array([np.linalg.inv(_) for _ in x])
    return grad * n_samples[:, None, None]


def _J(x, beta, alpha, gamma, lamda, S, n_samples):
    """Grad + prox + line search for the new point."""
    grad = grad_loss(x, S, n_samples)
    prox = prox_FL(x - gamma * grad, beta * gamma, alpha * gamma)
    return x + lamda * (prox - x)


def choose_lamda(lamda, x, emp_cov, n_samples, beta, alpha, gamma, delta=1e-4,
                 eps=0.5, max_iter=1000, criterion='a'):
    """Choose alpha for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741
    """
    # lamda = 1.
    partial_J = partial(_J, x, beta=beta, alpha=alpha, gamma=gamma, S=emp_cov,
                        n_samples=n_samples)
    partial_f = partial(loss, n_samples=n_samples, S=emp_cov)
    fx = partial_f(K=x)
    gradx = grad_loss(x, emp_cov, n_samples)
    gx = penalty(x, lamda, beta, l1_norm)
    for i in range(max_iter):
        x1 = partial_J(lamda=lamda)
        iter_diff = x1 - x
        loss_diff = partial_f(K=x1) - fx
        iter_diff_gradient = iter_diff.ravel().dot(gradx.ravel())

        if criterion == 'a':
            tolerance = delta * np.linalg.norm(iter_diff) / (gamma * lamda)
            gradx1 = grad_loss(x1, emp_cov, n_samples)
            grad_diff = gradx1.ravel() - gradx.ravel()
            if np.linalg.norm(grad_diff) <= tolerance:
                # print("Choose lamda = %.2f" % lamda)
                return lamda
        elif criterion == 'b':
            tolerance = delta * squared_norm(iter_diff) / (gamma * lamda)
            if loss_diff - iter_diff_gradient <= tolerance:
                # print("Choose lamda = %.2f" % lamda)
                return lamda
        elif criterion == 'c':
            obj_diff = objective(
                n_samples, emp_cov, x1, lamda, beta, l1_norm) - \
                objective(n_samples, emp_cov, x, lamda, beta, l1_norm)
            y = _J(x, beta, alpha, gamma, 1, emp_cov, n_samples)
            gy = penalty(y, lamda, beta, l1_norm)
            tolerance = (1 - delta) * lamda * (
                gy - gx + (y - x).ravel().dot(gradx.ravel()))
            if obj_diff <= tolerance:
                # print("Choose lamda = %.2f" % lamda)
                return lamda
        else:
            raise ValueError(criterion)
        lamda *= eps
    return lamda


def fista_step(Y, Y_diff, t):
    t_next = (1. + np.sqrt(1.0 + 4.0 * t*t)) / 2.
    return Y + ((t - 1.0)/t_next) * Y_diff, t_next


def time_graph_lasso(
        data_list, alpha=1., beta=1., max_iter=100, verbose=False,
        tol=1e-4, delta=1e-4, gamma=1.,
        return_history=False, return_n_iter=True,
        lamda_criterion='b'):
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
    emp_cov = np.array(list(map(empirical_covariance, data_list)))
    n_samples = np.array([s.shape[0] for s in data_list])
    # n_samples = np.array([1. for s in data_list])

    K = np.array([np.eye(s.shape[0]) for s in emp_cov])
    Y = K.copy()

    checks = []
    lamda = 1
    t = 1
    for iteration_ in range(max_iter):
        K_old = K.copy()  # np.ones_like(S) + 5000
        Y_old = Y.copy()

        # choose a gamma
        gamma = update_gamma(gamma, iteration_)

        # total variation
        # Y = _J(K, beta, alpha, gamma, 1, S, n_samples)
        Y = prox_FL(K - gamma * grad_loss(K, emp_cov, n_samples),
                    beta * gamma, alpha * gamma)

        lamda_n = choose_lamda(lamda, K, emp_cov, n_samples, beta, alpha, gamma,
                             delta=delta, criterion=lamda_criterion,
                             max_iter=50)
        K += lamda_n * (Y - K)
        # K = K + choose_lamda(lamda, K, emp_cov, n_samples, beta, alpha, gamma,
        #                      delta=delta, criterion=lamda_criterion,
        #                      max_iter=50) * (Y - K)

        # K, t = fista_step(Y, Y - Y_old, t)

        check = convergence(
            obj=objective(n_samples, emp_cov, K, alpha, beta, l1_norm),
            rnorm=np.linalg.norm(K - K_old),
            snorm=np.abs(objective(n_samples, emp_cov, K, alpha, beta, l1_norm) -
                         objective(n_samples, emp_cov, K_old, alpha, beta, l1_norm)),
            e_pri=tol, e_dual=tol)

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
