"""Useful proximal functions."""
import numpy as np
import warnings

from scipy.optimize import minimize

try:
    from prox_tv import tv1_1d
except:
    # fused lasso prox cannot be used
    pass


def soft_thresholding(a, lamda):
    """Soft-thresholding for vectors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.maximum(0, 1 - lamda / np.linalg.norm(a)) * a


def soft_thresholding_sign(a, lamda):
    """Soft-thresholding."""
    return np.sign(a) * np.maximum(np.abs(a) - lamda, 0)


def soft_thresholding_od(a, lamda):
    """Off-diagonal soft-thresholding."""
    soft = np.sign(a) * np.maximum(np.abs(a) - lamda, 0)
    soft.flat[::a.shape[1] + 1] = np.diag(a)
    return soft


def blockwise_soft_thresholding(a, lamda):
    """Proximal operator for l2 norm."""
    x = np.zeros_like(a)
    for t in range(a.shape[0]):
        x[t] = np.array([soft_thresholding(
            a[t, :, j], lamda) for j in range(a.shape[1])]).T
    return x


def prox_linf_1d(a, lamda):
    """Proximal operator for the l-inf norm.

    Since there is no closed-form, we can minimize it with scipy.
    """
    def _f(x):
        return lamda * np.linalg.norm(x, np.inf) + \
            .5 * np.power(np.linalg.norm(a - x), 2)
    return minimize(_f, a).x


def prox_linf(a, lamda):
    """Proximal operator for l-inf norm."""
    x = np.zeros_like(a)
    for t in range(a.shape[0]):
        x[t] = np.array([prox_linf_1d(
            a[t, :, j], lamda) for j in range(a.shape[1])]).T
    return x


def prox_logdet(a, lamda):
    """Time-varying latent variable graphical lasso prox."""
    es, Q = np.linalg.eigh(a)
    xi = (- es + np.sqrt(np.square(es) + 4. / lamda)) * lamda / 2.
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))


def prox_trace_indicator(a, lamda):
    """Time-varying latent variable graphical lasso prox."""
    es, Q = np.linalg.eigh(a)
    xi = np.maximum(es - lamda, 0)
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))


def prox_laplacian(a, lamda):
    """Prox for l_2 square norm, Laplacian regularisation."""
    return a / (1 + 2. * lamda)


def prox_FL(X, beta, lamda):
    """Fused Lasso prox.

    It is calculated as the Total variation prox + soft thresholding
    on the solution, as in
    http://ieeexplore.ieee.org/abstract/document/6579659/
    """
    Y = np.empty_like(X)
    for i in range(np.power(X.shape[1], 2)):
        solution = tv1_1d(X.flat[i::np.power(X.shape[1], 2)], beta)
        # fused-lasso (soft-thresholding on the solution)
        solution = soft_thresholding_sign(solution, lamda)
        Y.flat[i::np.power(X.shape[1], 2)] = solution
    return Y
