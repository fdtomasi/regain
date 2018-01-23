"""Useful proximal functions."""
import numpy as np
import warnings

from six.moves import range, zip
from scipy.optimize import minimize
from sklearn.utils.extmath import squared_norm

from regain.update_rules import update_rho
from regain.utils import convergence

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
            a[t, :, j], lamda) for j in range(a.shape[2])]).T
    return x


def blockwise_soft_thresholding_symmetric(a, lamda):
    """Proximal operator for l2 norm, for symmetric matrices (last 2 axes)."""
    col_norms = np.linalg.norm(a, axis=1)
    output = np.empty_like(a)
    for i, (x, c_norm) in enumerate(zip(a, col_norms)):
        output[i] = np.dot(x, np.diag(
            (np.ones(x.shape[0]) - lamda / c_norm) * (c_norm > lamda)))
    return output


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


def prox_node_penalty(A_12, lamda, rho=1, tol=1e-4, rtol=1e-2, max_iter=500):
    """Lamda = beta / (2. * rho).

    A_12 = np.vstack((A_1, A_2))
    """
    n_time, _, n_dim = A_12.shape

    U_1 = np.full((A_12.shape[0], n_dim, n_dim), 1. / n_dim, dtype=float)
    U_2 = np.copy(U_1)
    Y_1 = np.copy(U_1)
    Y_2 = np.copy(U_1)

    C = np.hstack((np.eye(n_dim), -np.eye(n_dim), np.eye(n_dim)))
    inverse = np.linalg.inv(C.T.dot(C) + 2 * np.eye(3 * n_dim))

    V = np.zeros_like(U_1)
    W = np.zeros_like(U_1)
    V_old = np.zeros_like(U_1)
    W_old = np.zeros_like(U_1)

    for iteration_ in range(max_iter):
        A = (Y_1 - Y_2 - W - U_1 + (W.transpose(0, 2, 1) - U_2).transpose(0, 2, 1)) / 2.
        V = blockwise_soft_thresholding_symmetric(A, lamda=lamda)

        A = np.concatenate(((V + U_2).transpose(0, 2, 1), A_12), axis=1)
        D = V + U_1
        # Z = np.linalg.solve(C.T*C + eta*np.identity(3*n), - C.T*D + eta* A)
        Z = np.empty_like(A)
        for i, (A_i, D_i) in enumerate(zip(A, D)):
            Z[i] = inverse.dot(2 * A_i - C.T.dot(D_i))
        W, Y_1, Y_2 = (Z[:, i*n_dim:(i+1) * n_dim, :] for i in range(3))

        # update residuals
        delta_U_1 = V + W - (Y_1 - Y_2)
        delta_U_2 = V - W.transpose(0, 2, 1)
        U_1 += delta_U_1
        U_2 += delta_U_2

        # diagnostics
        rnorm = np.sqrt(squared_norm(delta_U_1) + squared_norm(delta_U_2))
        snorm = rho * np.sqrt(squared_norm(W - W_old) +
                              squared_norm(V + W - V_old - W_old))
        check = convergence(
            obj=np.nan, rnorm=rnorm, snorm=snorm,
            e_pri=np.sqrt(2 * V.size) * tol + rtol * max(
                np.sqrt(squared_norm(W) + squared_norm(V + W)),
                np.sqrt(squared_norm(V) + squared_norm(Y_1 - Y_2))),
            e_dual=np.sqrt(2 * V.size) * tol + rtol * rho * np.sqrt(
                squared_norm(U_1) + squared_norm(U_2)))
        W_old = W.copy()
        V_old = V.copy()

        # if np.linalg.norm(delta_U_1, 'fro') < tol and \
        #         np.linalg.norm(delta_U_2, 'fro') < tol:
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_)
        # scaled dual variables should be also rescaled
        U_1 *= rho / rho_new
        U_2 *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Node norm did not converge.")
    return Y_1, Y_2


def prox_FL(a, beta, lamda):
    """Fused Lasso prox.

    It is calculated as the Total variation prox + soft thresholding
    on the solution, as in
    http://ieeexplore.ieee.org/abstract/document/6579659/
    """
    Y = np.empty_like(a)
    for i in range(np.power(a.shape[1], 2)):
        solution = tv1_1d(a.flat[i::np.power(a.shape[1], 2)], beta)
        # fused-lasso (soft-thresholding on the solution)
        solution = soft_thresholding_sign(solution, lamda)
        Y.flat[i::np.power(a.shape[1], 2)] = solution
    return Y

    # not work
    # x = np.vstack(a.transpose((2, 1, 0)))  # each row is the time evolution
    # assert np.allclose(np.array(xx), x)
    # x = np.apply_along_axis(
    #     compose(partial(soft_thresholding_sign, lamda=lamda),
    #             partial(tv1_1d, w=beta)), 1, x)
    # # return np.array(np.vsplit(x, x.shape[1])).transpose((2, 1, 0))
    # Z = np.array(np.vsplit(x, a.shape[1]))#.transpose((2, 1, 0))
