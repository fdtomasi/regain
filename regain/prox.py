# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Proximal functions."""
import warnings
from functools import partial

import collections

import numpy as np
from six.moves import range, zip
from sklearn.utils.extmath import squared_norm

from regain.update_rules import update_rho
from regain.utils import convergence

try:
    from prox_tv import tv1_1d, tvp_1d, tvgen, tvp_2d
except:
    # fused lasso prox cannot be used
    pass


def soft_thresholding(a, lamda):
    """Soft-thresholding."""
    return np.sign(a) * np.maximum(np.abs(a) - lamda, 0)


def _soft_thresholding_od_2d(a, lamda):
    # this assumes array is 2-dimensional
    # no check is performed for optimisation
    soft = soft_thresholding(a, lamda)
    np.fill_diagonal(soft, np.diag(a))
    return soft


def soft_thresholding_od(a, lamda):
    """Off-diagonal soft-thresholding."""
    if a.ndim > 2:
        out = np.empty_like(a)
        if not isinstance(lamda, collections.Iterable):
            lamda = np.repeat(lamda, a.shape[0])
        else:
            assert lamda.shape[0] == a.shape[0]

        for t in range(a.shape[0]):
            out[t] = _soft_thresholding_od_2d(a[t], lamda[t])
    else:
        out = _soft_thresholding_od_2d(a, lamda)
    return out


def soft_thresholding_vector(a, lamda):
    """Soft-thresholding for vectors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.maximum(1 - lamda / np.linalg.norm(a), 0) * a


def _blockwise_soft_thresholding_2d(a, lamda):
    """Proximal operator for l2 norm, a is two-dimensional."""
    return np.array([soft_thresholding_vector(aa, lamda) for aa in a.T]).T


def blockwise_soft_thresholding(a, lamda):
    """Proximal operator for l2 norm."""
    if a.ndim > 2:
        out = np.empty_like(a, dtype=float)
        if not isinstance(lamda, collections.Iterable):
            lamda = np.repeat(lamda, a.shape[0])
        else:
            lamda = lamda.ravel()
            assert lamda.shape[0] == a.shape[0]

        for t in range(a.shape[0]):
            out[t] = _blockwise_soft_thresholding_2d(a[t], lamda[t])
    else:
        out = _blockwise_soft_thresholding_2d(a, lamda)
    return out


def blockwise_soft_thresholding_symmetric(a, lamda):
    """Proximal operator for l2 norm, for symmetric matrices (last 2 axes)."""
    col_norms = np.linalg.norm(a, axis=1)
    ones_vect = np.ones(a.shape[1])

    if a.ndim > 2:
        out = np.empty_like(a, dtype=float)
        if not isinstance(lamda, collections.Iterable):
            lamda = np.repeat(lamda, a.shape[0])
        else:
            lamda = lamda.ravel()
            assert lamda.shape[0] == a.shape[0]

        out = np.empty_like(a, dtype=float)
        for t, (x, c_norm) in enumerate(zip(a, col_norms)):
            out[t] = np.dot(x, np.diag((ones_vect - lamda[t] / c_norm) * (c_norm > lamda[t])))

    else:
        out = np.dot(a, np.diag((ones_vect - lamda / col_norms) * (col_norms > lamda)))

    return out


# %% Perform prox operator:   min_x (1/2t)||x-w||^2 subject to |x|<=radius
# function [ z ] = project_1ball( z,radius )
# %  By Moreau's identity, projection onto 1-norm ball can be computed
# %  using the proximal of the conjugate problem, which is L-infinity
# %  minimization.
#   z = z -  prox_infinityNorm(z,radius);
# end
# %% Perform prox operator:   min ||x||_inf + (1/2t)||x-w||^2
# function [ xk ] = prox_infinityNorm( w,t )
#     N = length(w);
#     wabs = abs(w);
#     ws = (cumsum(sort(wabs,'descend'))- t)./(1:N)';
#     alphaopt = max(ws);
#     if alphaopt>0
#       xk = min(wabs,alphaopt).*sign(w); % truncation step
#     else
#       xk = zeros(size(w)); % if t is big, then solution is zero
#     end
# end
def prox_linf_1d(a, lamda):
    """Proximal operator for the l-inf norm.

    Since there is no closed-form, we can minimize it with scipy.
    """
    from scipy.optimize import minimize

    def _f(x):
        return lamda * np.linalg.norm(x, np.inf) + 0.5 * np.power(np.linalg.norm(a - x), 2)

    return minimize(_f, a).x


def prox_linf(a, lamda):
    """Proximal operator for l-inf norm."""
    x = np.zeros_like(a)
    for t in range(a.shape[0]):
        x[t] = np.array([prox_linf_1d(a[t, :, j], lamda) for j in range(a.shape[1])]).T
    return x


def prox_logdet(a, lamda):
    """Time-varying latent variable graphical lasso prox."""
    es, Q = np.linalg.eigh(a)
    xi = (-es + np.sqrt(np.square(es) + 4.0 / lamda)) * lamda / 2.0
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))


def prox_logdet_ala_ma(a, lamda):
    es, Q = np.linalg.eigh(a)
    xi = (-es + np.sqrt(np.square(es) + 4.0 * lamda)) / 2.0
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))


def prox_trace_indicator(a, lamda):
    """Time-varying latent variable graphical lasso prox."""
    es, Q = np.linalg.eigh(a)
    xi = np.maximum(es - lamda, 0)
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))


def prox_laplacian(a, lamda):
    """Prox for l_2 square norm, Laplacian regularisation."""
    return a / (1 + 2.0 * lamda)


def prox_node_penalty(A_12, lamda, rho=1, tol=1e-4, rtol=1e-2, max_iter=500):
    """Lamda = beta / (2. * rho).

    A_12 = np.vstack((A_1, A_2))
    """
    n_time, _, n_dim = A_12.shape

    U_1 = np.full((A_12.shape[0], n_dim, n_dim), 1.0 / n_dim, dtype=float)
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
        A = (Y_1 - Y_2 - W - U_1 + (W.transpose(0, 2, 1) - U_2).transpose(0, 2, 1)) / 2.0
        V = blockwise_soft_thresholding_symmetric(A, lamda=lamda)

        A = np.concatenate(((V + U_2).transpose(0, 2, 1), A_12), axis=1)
        D = V + U_1
        # Z = np.linalg.solve(C.T*C + eta*np.identity(3*n), - C.T*D + eta* A)
        Z = np.empty_like(A)
        for i, (A_i, D_i) in enumerate(zip(A, D)):
            Z[i] = inverse.dot(2 * A_i - C.T.dot(D_i))
        W, Y_1, Y_2 = (Z[:, i * n_dim : (i + 1) * n_dim, :] for i in range(3))

        # update residuals
        delta_U_1 = V + W - (Y_1 - Y_2)
        delta_U_2 = V - W.transpose(0, 2, 1)
        U_1 += delta_U_1
        U_2 += delta_U_2

        # diagnostics
        rnorm = np.sqrt(squared_norm(delta_U_1) + squared_norm(delta_U_2))
        snorm = rho * np.sqrt(squared_norm(W - W_old) + squared_norm(V + W - V_old - W_old))
        check = convergence(
            obj=np.nan,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=np.sqrt(2 * V.size) * tol
            + rtol
            * max(np.sqrt(squared_norm(W) + squared_norm(V + W)), np.sqrt(squared_norm(V) + squared_norm(Y_1 - Y_2))),
            e_dual=np.sqrt(2 * V.size) * tol + rtol * rho * np.sqrt(squared_norm(U_1) + squared_norm(U_2)),
        )
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


def prox_FL(a, beta, lamda, p=1, symmetric=False, use_matlab=False, optimize=True):
    """Fused Lasso prox.

    It is calculated as the Total variation prox + soft thresholding
    on the solution, as in
    http://ieeexplore.ieee.org/abstract/document/6579659/
    """
    # if any([any(np.diag(x) < 0) for x in a]):
    #     for a_i in a:
    #         np.fill_diagonal(a_i, np.sum(np.abs(a_i), axis=1))
    if optimize:
        Y = tvgen(a, [beta], [1], [p], n_threads=32, max_iters=30)

    else:
        Y = np.empty_like(a)
        # if use_matlab:
        #     from regain.wrapper.tv_condat import wrapper
        #     func = wrapper.total_variation_condat
        # else:
        func = tv1_1d if p == 1 else partial(tvp_1d, p=p)

        if symmetric:
            x, y = np.triu_indices_from(a[0])
            b = np.vstack(a.transpose(1, 2, 0))
            upper_ind = x * a.shape[1] + y
            Z = np.zeros_like(b)
            Z[upper_ind] = [func(row, beta) for row in b[upper_ind]]

            e = np.array(np.split(Z, a.shape[1], axis=0)).transpose(2, 0, 1)
            Y = (e + e.transpose(0, 2, 1)) / (np.array([np.eye(a.shape[1]) for i in range(a.shape[0])]) + 1)
        else:
            for i in range(np.power(a.shape[1], 2)):
                solution = func(a.flat[i :: np.power(a.shape[1], 2)], beta)
                Y.flat[i :: np.power(a.shape[1], 2)] = solution

    # fused-lasso (soft-thresholding on the solution)
    Y_soft = soft_thresholding(Y, lamda)

    # restore diagonal
    for y_s, y_fv in zip(Y_soft, Y):
        np.fill_diagonal(y_s, np.diag(y_fv))
    return Y_soft

    # Y = np.empty_like(a)
    # for i in range(a.shape[1]):
    #     for j in range(i, a.shape[2]):
    #         solution = tv1_1d(a[:, i, j], beta)
    #         # fused-lasso (soft-thresholding on the solution)
    #         if i != j:
    #             solution = soft_thresholding_sign(solution, lamda)
    #         Y[:, i, j] = solution
    # return Y

    # not work
    # x = np.vstack(a.transpose((2, 1, 0)))  # each row is the time evolution
    # assert np.allclose(np.array(xx), x)
    # x = np.apply_along_axis(
    #     compose(partial(soft_thresholding_sign, lamda=lamda),
    #             partial(tv1_1d, w=beta)), 1, x)
    # # return np.array(np.vsplit(x, x.shape[1])).transpose((2, 1, 0))
    # Z = np.array(np.vsplit(x, a.shape[1]))#.transpose((2, 1, 0))
