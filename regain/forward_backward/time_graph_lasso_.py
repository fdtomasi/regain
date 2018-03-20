"""Time graph lasso via forward_backward (for now only in case of l1 norm)."""
from __future__ import division

import warnings
from functools import partial

import numpy as np
from six.moves import map, range, zip
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import squared_norm

from regain.covariance.graph_lasso_ import logl
from regain.norm import l1_norm, l1_od_norm
from regain.prox import prox_FL
from regain.utils import convergence


def loss_function(n_samples, S, K):
    return sum(-n * logl(emp_cov, precision)
               for emp_cov, precision, n in zip(S, K, n_samples))


def _gradient(x, emp_cov, n_samples):
    grad = emp_cov - np.array([np.linalg.inv(_) for _ in x])
    return grad * n_samples[:, None, None]


def _J(x, beta, alpha, gamma, lamda, S, n_samples):
    grad_ = _gradient(x, S, n_samples)
    prox_ = prox_FL(x - gamma * grad_, beta * gamma, alpha * gamma)
    return x + lamda * (prox_ - x)


def _scalar_product_3d(x, y):
    # ss = np.sum([xx.T.dot(yy) for xx, yy in zip(x, y)])
    # ss2 = np.tensordot(x.T, y, axes=1).sum()
    return x.ravel().dot(y.ravel())


def update_gamma(gamma, iteration):
    if iteration % 10 == 0:
        return gamma / 2.
    return gamma


def _g(precision, lamda, beta):
    obj = lamda * sum(map(l1_od_norm, precision))
    obj += beta * sum(map(l1_norm, precision[1:] - precision[:-1]))
    return obj


def objective(n_samples, emp_cov, precision, lamda, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = loss_function(n_samples, emp_cov, precision)
    obj += lamda * sum(map(l1_od_norm, precision))
    obj += beta * sum(map(psi, precision[1:] - precision[:-1]))
    return obj


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
    partial_f = partial(loss_function, n_samples=n_samples, S=emp_cov)
    fx = partial_f(K=x)
    gradx = _gradient(x, emp_cov, n_samples)
    gx = _g(x, lamda, beta)
    for i in range(max_iter):
        x_lamda = partial_J(lamda=lamda)
        iter_diff = x_lamda - x
        loss_diff = partial_f(K=x_lamda) - fx
        iter_diff_gradient = _scalar_product_3d(iter_diff, gradx)

        if criterion == 'a':
            tolerance = delta * np.linalg.norm(iter_diff) / (gamma * lamda)
            gradx1 = _gradient(x_lamda, emp_cov, n_samples)
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
                n_samples, emp_cov, x_lamda, lamda, beta, l1_norm) - \
                objective(n_samples, emp_cov, x, lamda, beta, l1_norm)
            y = _J(x, beta, alpha, gamma, 1, emp_cov, n_samples)
            gy = _g(y, lamda, beta)
            tolerance = (1 - delta) * lamda * (
                gy - gx + (y - x).ravel().dot(gradx.ravel()))
            if obj_diff <= tolerance:
                # print("Choose lamda = %.2f" % lamda)
                return lamda
        else:
            raise ValueError(criterion)
        lamda *= eps
    return lamda


def time_graph_lasso(
        data_list, alpha=1., beta=1., max_iter=100, verbose=False,
        tol=1e-4, delta=1e-4, gamma=1., return_history=False,
        lamda_criterion='a'):
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
    S = np.array(list(map(empirical_covariance, data_list)))
    # n_samples = np.array([s.shape[0] for s in data_list])
    n_samples = np.array([1 for s in data_list])

    K = np.array([np.eye(s.shape[0]) for s in S])

    checks = []
    lamda = 1
    K_old = K.copy()  # np.ones_like(S) + 5000
    for iteration_ in range(max_iter):
        lamda_old = lamda

        # choose a gamma
        gamma = update_gamma(gamma, iteration_)

        # total variation
        # Y = _J(K, beta, alpha, gamma, 1, S, n_samples)
        Y = prox_FL(K - gamma * _gradient(K, S, n_samples),
                    beta * gamma, alpha * gamma)

        lamda = choose_lamda(lamda_old, K, S, n_samples, beta, alpha, gamma,
                             delta=delta, criterion=lamda_criterion)
        # alpha = 1
        K += lamda * (Y - K)

        check = convergence(
            obj=objective(n_samples, S, K, alpha, beta, l1_norm),
            rnorm=np.linalg.norm(K - K_old),
            snorm=np.abs(objective(n_samples, S, K, alpha, beta, l1_norm) -
                         objective(n_samples, S, K_old, alpha, beta, l1_norm)),
            e_pri=tol,
            e_dual=tol
        )

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
        K_old = K.copy()
    else:
        warnings.warn("Objective did not converge.")

    if return_history:
        return K, S, checks
    return K, S
