# BSD 3-Clause License

# Copyright (c) 2017, Federico T.
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
"""Graphical latent variable model selection via ADMM."""
from __future__ import division

import warnings

import numpy as np
from scipy import linalg
from six.moves import range
from sklearn.utils.extmath import squared_norm

from regain.norm import l1_od_norm
from regain.prox import prox_laplacian, prox_trace_indicator, soft_thresholding
from regain.update_rules import update_rho
from regain.utils import convergence


def objective(S, R, K, L, alpha, tau):
    """Objective function for latent graphical lasso."""
    obj = 0.5 * squared_norm(S - R)
    obj += alpha * l1_od_norm(K)
    obj += tau * np.linalg.norm(L, ord="nuc")
    return obj


def infimal_convolution(
    S,
    alpha=1.0,
    tau=1.0,
    rho=1.0,
    max_iter=100,
    verbose=False,
    tol=1e-4,
    rtol=1e-2,
    return_history=False,
    return_n_iter=True,
    update_rho_options=None,
    compute_objective=True,
):
    r"""Latent variable graphical lasso solver.

    Solves the following problem via ADMM:
        min - log_likelihood(S, K-L) + alpha ||K||_{od,1} + tau ||L_i||_*

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    emp_cov : array-like
        Empirical covariance matrix.
    alpha, tau : float, optional
        Regularisation parameters.
    rho : float, optional
        Augmented Lagrangian parameter.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Absolute tolerance for convergence.
    rtol : float, optional
        Relative tolerance for convergence.
    return_history : bool, optional
        Return the history of computed values.
    return_n_iter : bool, optional
        Return the number of iteration before convergence.
    verbose : bool, default False
        Print info at each iteration.

    Returns
    -------
    K, L : np.array, 2-dimensional, size (d x d)
        Solution to the problem.
    S : np.array, 2 dimensional
        Empirical covariance matrix.
    n_iter : int
        If return_n_iter, returns the number of iterations before convergence.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    K = np.zeros_like(S)
    L = np.zeros_like(S)
    U = np.zeros_like(S)
    R_old = np.zeros_like(S)

    checks = []
    for iteration_ in range(max_iter):
        # update R
        A = K - L - U
        A += A.T
        A /= 2.0
        R = prox_laplacian(S + rho * A, lamda=rho / 2.0)

        A = L + R + U
        K = soft_thresholding(A, lamda=alpha / rho)

        A = K - R - U
        A += A.T
        A /= 2.0
        L = prox_trace_indicator(A, lamda=tau / rho)

        # update residuals
        U += R - K + L

        # diagnostics, reporting, termination checks
        obj = objective(S, R, K, L, alpha, tau) if compute_objective else np.nan
        rnorm = np.linalg.norm(R - K + L)
        snorm = rho * np.linalg.norm(R - R_old)
        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=np.sqrt(R.size) * tol + rtol * max(np.linalg.norm(R), np.linalg.norm(K - L)),
            e_dual=np.sqrt(R.size) * tol + rtol * rho * np.linalg.norm(U),
        )
        R_old = R.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f," "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
        if check.obj == np.inf:
            break
        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_, **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        U *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    covariance_ = linalg.pinvh(K)
    return_list = [K, L, covariance_]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list
