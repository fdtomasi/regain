"""Time graph lasso via forward_backward (for now only in case of l1 norm)."""
from __future__ import division, print_function

import warnings
from functools import partial

import numpy as np
from scipy import linalg
from six.moves import map, range, zip
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import squared_norm

from regain.covariance.graph_lasso_ import logl
from regain.covariance.time_graph_lasso_ import TimeGraphLasso
from regain.norm import l1_od_norm, vector_p_norm
from regain.prox import prox_FL, soft_thresholding, soft_thresholding_od
from regain.utils import convergence, positive_definite

from .time_graphical_lasso_ import loss as loss_tglfb
from .time_graphical_lasso_ import grad_loss as grad_loss_tglfb


def loss_laplacian(S, K, beta=0, n_samples=None, vareps=0):
    """Loss function for time-varying graphical lasso."""
    loss_ = loss_tglfb(S, K, n_samples=n_samples, vareps=vareps)
    loss_ += beta * squared_norm(K[1:] - K[:-1])
    return loss_


def grad_loss_laplacian(
        x, emp_cov, beta=0, n_samples=None, x_inv=None, vareps=0):
    """Gradient of the loss function for the time-varying graphical lasso."""
    grad = grad_loss_tglfb(x, emp_cov, n_samples, x_inv, vareps)

    aux = np.empty_like(x)
    aux[0] = x[0] - x[1]
    aux[-1] = x[-1] - x[-2]
    for t in range(1, x.shape[0] - 1):
        aux[t] = 2 * x[t] - x[t-1] - x[t+1]
    aux *= 2 * beta
    grad += aux

    return grad


def penalty_laplacian(precision, alpha):
    """Penalty for time-varying graphical lasso."""
    if isinstance(alpha, np.ndarray):
        obj = sum(a[0][0] * m for a, m in zip(alpha, map(l1_od_norm, precision)))
    else:
        obj = alpha * sum(map(l1_od_norm, precision))
    # obj += beta * psi(precision[1:] - precision[:-1])
    return obj


def prox_penalty_laplacian(precision, alpha):
    # return soft_thresholding(precision, alpha)
    return np.array([soft_thresholding_od(p, alpha) for p in precision])


def objective_laplacian(n_samples, emp_cov, precision, alpha, beta, vareps=0):
    """Objective function for time-varying graphical lasso."""
    obj = loss_laplacian(
        emp_cov, precision, beta=beta, n_samples=n_samples, vareps=vareps)
    obj += penalty_laplacian(precision, alpha)
    return obj




def choose_gamma_laplacian(
        gamma, x, emp_cov, n_samples, beta, alpha, lamda, grad, delta=1e-4,
        eps=0.5, max_iter=1000, p=1, x_inv=None, vareps=1e-5, choose='gamma'):
    """Choose gamma for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741

    """
    # if grad is None:
    #     grad = grad_loss(x, emp_cov, n_samples, x_inv=x_inv)

    partial_f = partial(
        loss_laplacian, beta=beta, n_samples=n_samples, S=emp_cov,
        vareps=vareps)
    fx = partial_f(K=x)
    for i in range(max_iter):
        prox = prox_penalty_laplacian(x - gamma * grad, alpha * gamma)
        if positive_definite(prox) and choose != "gamma":
            break

        if choose == "gamma":
            y_minus_x = prox - x
            loss_diff = partial_f(K=x + lamda * y_minus_x) - fx

            tolerance = _scalar_product(y_minus_x, grad)
            tolerance += delta / gamma * _scalar_product(y_minus_x, y_minus_x)
            if loss_diff <= lamda * tolerance:
                break
        gamma *= eps

    return gamma, prox


def choose_lamda_laplacian(
        lamda, x, emp_cov, n_samples, beta, alpha, gamma, delta=1e-4, eps=0.5,
        max_iter=1000, criterion='b', p=1, x_inv=None, grad=None, prox=None,
        min_eigen_x=None, vareps=1e-5):
    """Choose lambda for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741

    """
    # if x_inv is None:
    #     x_inv = np.array([linalg.pinvh(_) for _ in x])
    # if grad is None:
    #     grad = grad_loss(x, emp_cov, n_samples, x_inv=x_inv)
    # if prox is None:
    #     prox = prox_FL(x - gamma * grad, beta * gamma, alpha * gamma, p=p, symmetric=True)

    partial_f = partial(
        loss_laplacian, beta=beta, n_samples=n_samples, S=emp_cov,
        vareps=vareps)
    fx = partial_f(K=x)

    # min_eigen_y = np.min([np.linalg.eigh(z)[0] for z in prox])

    y_minus_x = prox - x
    if criterion == 'b':
        tolerance = _scalar_product(y_minus_x, grad)
        tolerance += delta / gamma * _scalar_product(y_minus_x, y_minus_x)
    elif criterion == 'c':
        psi = partial(vector_p_norm, p=p)
        gx = penalty_laplacian(x, alpha, beta, psi)
        gy = penalty_laplacian(prox, alpha, beta, psi)
        objective_x = objective_laplacian(
            n_samples, emp_cov, x, alpha, beta, psi, vareps=vareps)
        tolerance = (1 - delta) * (gy - gx + _scalar_product(y_minus_x, grad))

    for i in range(max_iter):
        # line-search
        x1 = x + lamda * y_minus_x

        if criterion == 'a':
            iter_diff = x1 - x
            gradx1 = grad_loss_laplacian(x1, emp_cov, n_samples)
            grad_diff = gradx1 - grad
            norm_grad_diff = np.sqrt(_scalar_product(grad_diff, grad_diff))
            norm_iter_diff = np.sqrt(_scalar_product(iter_diff, iter_diff))
            tolerance = delta * norm_iter_diff / (gamma * lamda)
            if norm_grad_diff <= tolerance:
                break
        elif criterion == 'b':
            loss_diff = partial_f(K=x1) - fx
            if loss_diff <= lamda * tolerance and positive_definite(x1):
                break
        elif criterion == 'c':
            obj_diff = objective_laplacian(
                n_samples, emp_cov, x1, alpha, beta, psi, vareps=vareps) \
                    - objective_x
            # if positive_definite(x1) and obj_diff <= lamda * tolerance:
            cond = True # lamda > 0 if min_eigen_y >= 0 else lamda < min_eigen_x / (min_eigen_x - min_eigen_y)
            if cond and obj_diff <= lamda * tolerance:
                break
        else:
            raise ValueError(criterion)
        lamda *= eps
    return lamda, i + 1




def time_graphical_lasso(
        emp_cov, n_samples, alpha=0.01, beta=1., max_iter=100, verbose=False,
        tol=1e-4, delta=1e-4, gamma=1., lamda=1., eps=0.5, debug=False,
        return_history=False, return_n_iter=True, choose='gamma',
        lamda_criterion='b', time_norm=1, compute_objective=True,
        return_n_linesearch=False, vareps=1e-5, stop_at=None, stop_when=1e-4):
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
    available_choose = ('gamma', 'lamda', 'fixed', 'both')
    if choose not in available_choose:
        raise ValueError("`choose` parameter must be one of %s." %
                         available_choose)

    n_times, _, n_features = emp_cov.shape
    covariance_ = emp_cov.copy()
    covariance_ *= 0.95

    K = np.empty_like(emp_cov)
    for i, (c, e) in enumerate(zip(covariance_, emp_cov)):
        c.flat[::n_features + 1] = e.flat[::n_features + 1]
        K[i] = linalg.pinvh(c)

    # K = np.array([np.eye(s.shape[0]) for s in emp_cov])
    # Y = K.copy()
    # assert positive_definite(K)

    obj_partial = partial(
        objective_laplacian, n_samples=n_samples, emp_cov=emp_cov, alpha=alpha,
        beta=beta, vareps=vareps)

    max_residual = -np.inf
    n_linesearch = 0
    checks = [convergence(obj=obj_partial(precision=K))]
    for iteration_ in range(max_iter):
        # if not positive_definite(K):
        #     print("precision is not positive definite.")
        #     break

        k_previous = K.copy()
        x_inv = np.array([linalg.pinvh(x) for x in K])
        # x_inv = []
        # eigens = []
        # for x in K:
        #     es, Q = np.linalg.eigh(x)
        #     Inv = np.linalg.multi_dot((Q, np.diag(1. / es), Q.T))
        #     x_inv.append(Inv)
        #     eigens.append(es)
        # x_inv = np.array(x_inv)
        # eigens = np.array(eigens)

        grad = grad_loss_laplacian(
            K, emp_cov, beta=beta, n_samples=n_samples, x_inv=x_inv,
            vareps=vareps)
        if choose in ['gamma', 'both']:
            gamma, y = choose_gamma(
                gamma / eps if iteration_ > 0 else gamma, K, emp_cov,
                n_samples=n_samples,
                beta=beta, alpha=alpha, lamda=lamda, grad=grad,
                delta=delta, eps=eps, max_iter=200, p=time_norm, x_inv=x_inv,
                vareps=vareps, choose=choose)
            # gamma = min(gamma, 0.249)
        # print(gamma)

        x_hat = K - gamma * grad
        if choose not in ['gamma', 'both']:
            y = prox_penalty_laplacian(x_hat, alpha * gamma)

        if choose in ['lamda', 'both']:
            lamda, n_ls = choose_lamda(
                min(lamda / eps if iteration_ > 0 else lamda, 1),
                K, emp_cov, n_samples=n_samples,
                beta=beta, alpha=alpha, gamma=gamma, delta=delta, eps=eps,
                criterion=lamda_criterion, max_iter=200, p=time_norm,
                x_inv=x_inv, grad=grad, prox=y,
                # min_eigen_x=np.min(eigens),
                vareps=vareps)
            n_linesearch += n_ls
        # print ("lambda: ", lamda, n_ls)

        K = K + min(max(lamda, 0), 1) * (y - K)
        # K, t = fista_step(Y, Y - Y_old, t)

        check = convergence(
            obj=obj_partial(precision=K),
            rnorm=np.linalg.norm(upper_diag_3d(K) - upper_diag_3d(k_previous)),
            snorm=np.linalg.norm(
                obj_partial(precision=K) - obj_partial(precision=k_previous)),
            e_pri=np.sqrt(upper_diag_3d(K).size) * tol + tol * max(
                np.linalg.norm(upper_diag_3d(K)),
                np.linalg.norm(upper_diag_3d(k_previous))),
            e_dual=tol,
            # precision=K.copy()
            )

        if verbose and iteration_ % (50 if verbose < 2 else 1) == 0:
            print("obj: %.4f, rnorm: %.7f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        if return_history:
            checks.append(check)

        if np.isnan(check.rnorm) or np.isnan(check.snorm):
            warnings.warn("precision is not positive definite.")

        if stop_at is not None:
            if abs(check.obj - stop_at) / abs(stop_at) < stop_when:
                break

        # use this convergence criterion
        subgrad = (x_hat - K) / gamma
        if 0:
            grad = grad_loss_laplacian(K, emp_cov, n_samples, vareps=vareps)
            res_norm = np.linalg.norm(grad + subgrad)

            if iteration_ == 0:
                normalizer = res_norm + 1e-6
            max_residual = max(np.linalg.norm(grad),
                               np.linalg.norm(subgrad)) + 1e-6
        else:
            res_norm = np.linalg.norm(K - k_previous) / gamma
            max_residual = max(max_residual, res_norm)
            normalizer = max(np.linalg.norm(grad),
                             np.linalg.norm(subgrad)) + 1e-6

        r_rel = res_norm / max_residual
        r_norm = res_norm / normalizer

        if not debug and (r_rel <= tol or r_norm <= tol) and iteration_ > 0: # or (
            # check.rnorm <= check.e_pri and iteration_ > 0):
            break
            # pass
        # if check.rnorm <= check.e_pri and iteration_ > 0:
        #     # and check.snorm <= check.e_dual:
        #     break
    else:
        warnings.warn("Objective did not converge.")

    # for i in range(K.shape[0]):
    #     covariance_[i] = linalg.pinvh(K[i])

    return_list = [K, covariance_]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_ + 1)
    if return_n_linesearch:
        return_list.append(n_linesearch)
    return return_list
