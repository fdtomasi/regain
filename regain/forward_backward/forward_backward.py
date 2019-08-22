import numpy as np
from regain.prox import soft_thresholding_od, prox_FL
from regain.utils import positive_definite


def _scalar_product(x, y):
    return (x * y).sum()
    # return x.ravel().dot(y.ravel())


def fista_step(Y, Y_diff, t):
    t_next = (1. + np.sqrt(1.0 + 4.0 * t * t)) / 2.
    return Y + ((t - 1.0) / t_next) * Y_diff, t_next


def upper_diag_3d(x):
    """Return the flattened upper diagonal of a 3d matrix."""
    # n_times, n_dim, _ = x.shape
    # upper_idx = np.triu_indices(n_dim, 1)
    # return np.array([xx[upper_idx] for xx in x])
    return np.triu(x, 1)


def choose_gamma(
        gamma, x, beta, alpha, lamda, grad, function_f=None, delta=1e-4,
        eps=0.5, max_iter=1000, p=1, x_inv=None, choose='gamma',
        laplacian_penalty=False):
    """Choose gamma for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741

    """
    fx = function_f(K=x)
    for i in range(max_iter):
        if laplacian_penalty:
            prox = soft_thresholding_od(x - gamma * grad, alpha * gamma)
        else:
            prox = prox_FL(
                x - gamma * grad, beta * gamma, alpha * gamma, p=p,
                symmetric=True)
        if positive_definite(prox) and choose != "gamma":
            break

        if choose == "gamma":
            y_minus_x = prox - x
            loss_diff = function_f(K=x + lamda * y_minus_x) - fx

            tolerance = _scalar_product(y_minus_x, grad)
            tolerance += delta / gamma * _scalar_product(y_minus_x, y_minus_x)
            if loss_diff <= lamda * tolerance:
                break
        gamma *= eps

    return gamma, prox


def choose_lamda(
        lamda, x, gamma, delta=1e-4, eps=0.5, function_f=None, penalty_f=None,
        objective_f=None, gradient_f=None, function_g=None, max_iter=1000,
        criterion='b', p=1, grad=None, prox=None, min_eigen_x=None):
    """Choose lambda for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741

    """
    fx = function_f(K=x)
    # min_eigen_y = np.min([np.linalg.eigh(z)[0] for z in prox])

    y_minus_x = prox - x
    if criterion == 'b':
        tolerance = _scalar_product(y_minus_x, grad)
        tolerance += delta / gamma * _scalar_product(y_minus_x, y_minus_x)
    elif criterion == 'c':
        objective_x = objective_f(x)
        gx = function_g(x)
        gy = function_g(prox)
        tolerance = (1 - delta) * (gy - gx + _scalar_product(y_minus_x, grad))

    for i in range(max_iter):
        # line-search
        x1 = x + lamda * y_minus_x

        if criterion == 'a':
            iter_diff = x1 - x
            gradx1 = gradient_f(x1)
            grad_diff = gradx1 - grad
            norm_grad_diff = np.sqrt(_scalar_product(grad_diff, grad_diff))
            norm_iter_diff = np.sqrt(_scalar_product(iter_diff, iter_diff))
            tolerance = delta * norm_iter_diff / (gamma * lamda)
            if norm_grad_diff <= tolerance:
                break
        elif criterion == 'b':
            loss_diff = function_f(K=x1) - fx
            if loss_diff <= lamda * tolerance and positive_definite(x1):
                break
        elif criterion == 'c':
            obj_diff = objective_f(x1) - objective_x
            # if positive_definite(x1) and obj_diff <= lamda * tolerance:
            cond = True
            # COND IS lamda > 0 if min_eigen_y >= 0 else lamda < min_eigen_x / (min_eigen_x - min_eigen_y)
            if cond and obj_diff <= lamda * tolerance:
                break
        else:
            raise ValueError(criterion)
        lamda *= eps
    return lamda, i + 1
