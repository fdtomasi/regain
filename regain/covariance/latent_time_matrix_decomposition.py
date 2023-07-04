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
"""Latent variable time-varying matrix decomposition using ADMM."""
from __future__ import division

import warnings

import numpy as np
from six.moves import map, range, zip
from sklearn.utils.extmath import squared_norm

from regain.covariance.latent_time_graphical_lasso_ import LatentTimeGraphicalLasso
from regain.norm import l1_od_norm
from regain.prox import prox_trace_indicator, soft_thresholding
from regain.update_rules import update_rho
from regain.utils import Convergence
from regain.validation import check_input, check_norm_prox


def objective(S, R, Z_0, Z_1, Z_2, W_0, W_1, W_2, alpha, tau, beta, eta, psi, phi):
    """Objective for latent variable time-varying matrix decomposition."""
    obj = squared_norm(S - R)
    obj += alpha * np.sum(l1_od_norm(Z_0))
    obj += tau * np.sum(np.linalg.norm(W_0, ord="nuc", axis=(-2, -1)))
    obj += beta * sum(map(psi, Z_2 - Z_1))
    obj += eta * sum(map(phi, W_2 - W_1))
    return obj


def latent_time_matrix_decomposition(
    emp_cov,
    alpha=0.01,
    tau=1.0,
    rho=1.0,
    beta=1.0,
    eta=1.0,
    max_iter=100,
    verbose=False,
    psi="laplacian",
    phi="laplacian",
    mode="admm",
    tol=1e-4,
    rtol=1e-4,
    assume_centered=False,
    return_history=False,
    return_n_iter=True,
    update_rho_options=None,
    compute_objective=True,
):
    r"""Latent variable time-varying matrix decomposition solver.

    Solves the following problem via ADMM:
        min sum_{i=1}^T || S_i-(K_i-L_i)||^2 + alpha ||K_i||_{od,1}
            + tau ||L_i||_*
            + beta sum_{i=2}^T Psi(K_i - K_{i-1})
            + eta sum_{i=2}^T Phi(L_i - L_{i-1})

    where S is the matrix to decompose.

    Parameters
    ----------
    emp_cov : ndarray, shape (n_features, n_features)
        Matrix to decompose.
    alpha, tau, beta, eta : float, optional
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

    Returns
    -------
    K, L : numpy.array, 3-dimensional (T x d x d)
        Solution to the problem for each time t=1...T .
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    psi, prox_psi, psi_node_penalty = check_norm_prox(psi)
    phi, prox_phi, phi_node_penalty = check_norm_prox(phi)

    Z_0 = np.zeros_like(emp_cov)
    Z_1 = np.zeros_like(Z_0)[:-1]
    Z_2 = np.zeros_like(Z_0)[1:]
    W_0 = np.zeros_like(Z_0)
    W_1 = np.zeros_like(Z_1)
    W_2 = np.zeros_like(Z_2)

    X_0 = np.zeros_like(Z_0)
    X_1 = np.zeros_like(Z_1)
    X_2 = np.zeros_like(Z_2)
    U_1 = np.zeros_like(W_1)
    U_2 = np.zeros_like(W_2)

    R_old = np.zeros_like(Z_0)
    Z_1_old = np.zeros_like(Z_1)
    Z_2_old = np.zeros_like(Z_2)
    W_1_old = np.zeros_like(W_1)
    W_2_old = np.zeros_like(W_2)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.full(emp_cov.shape[0], 3, dtype=float)
    divisor[0] -= 1
    divisor[-1] -= 1

    checks = []
    for iteration_ in range(max_iter):
        # update R
        A = Z_0 - W_0 - X_0
        R = (rho * A + 2 * emp_cov) / (2 + rho)

        # update Z_0
        A = R + W_0 + X_0
        A[:-1] += Z_1 - X_1
        A[1:] += Z_2 - X_2
        A /= divisor[:, None, None]
        # soft_thresholding_ = partial(soft_thresholding, lamda=alpha / rho)
        # Z_0 = np.array(map(soft_thresholding_, A))
        Z_0 = soft_thresholding(A, lamda=alpha / (rho * divisor[:, None, None]))

        # update Z_1, Z_2
        A_1 = Z_0[:-1] + X_1
        A_2 = Z_0[1:] + X_2
        if not psi_node_penalty:
            prox_e = prox_psi(A_2 - A_1, lamda=2.0 * beta / rho)
            Z_1 = 0.5 * (A_1 + A_2 - prox_e)
            Z_2 = 0.5 * (A_1 + A_2 + prox_e)
        else:
            Z_1, Z_2 = prox_psi(
                np.concatenate((A_1, A_2), axis=1),
                lamda=0.5 * beta / rho,
                rho=rho,
                tol=tol,
                rtol=rtol,
                max_iter=max_iter,
            )

        # update W_0
        A = Z_0 - R - X_0
        A[:-1] += W_1 - U_1
        A[1:] += W_2 - U_2
        A /= divisor[:, None, None]
        A += A.transpose(0, 2, 1)
        A /= 2.0

        W_0 = np.array(
            [
                prox_trace_indicator(a, lamda=tau / (rho * div))
                for a, div in zip(A, divisor)
            ]
        )

        # update W_1, W_2
        A_1 = W_0[:-1] + U_1
        A_2 = W_0[1:] + U_2
        if not phi_node_penalty:
            prox_e = prox_phi(A_2 - A_1, lamda=2.0 * eta / rho)
            W_1 = 0.5 * (A_1 + A_2 - prox_e)
            W_2 = 0.5 * (A_1 + A_2 + prox_e)
        else:
            W_1, W_2 = prox_phi(
                np.concatenate((A_1, A_2), axis=1),
                lamda=0.5 * eta / rho,
                rho=rho,
                tol=tol,
                rtol=rtol,
                max_iter=max_iter,
            )

        # update residuals
        X_0 += R - Z_0 + W_0
        X_1 += Z_0[:-1] - Z_1
        X_2 += Z_0[1:] - Z_2
        U_1 += W_0[:-1] - W_1
        U_2 += W_0[1:] - W_2

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(R - Z_0 + W_0)
            + squared_norm(Z_0[:-1] - Z_1)
            + squared_norm(Z_0[1:] - Z_2)
            + squared_norm(W_0[:-1] - W_1)
            + squared_norm(W_0[1:] - W_2)
        )

        snorm = rho * np.sqrt(
            squared_norm(R - R_old)
            + squared_norm(Z_1 - Z_1_old)
            + squared_norm(Z_2 - Z_2_old)
            + squared_norm(W_1 - W_1_old)
            + squared_norm(W_2 - W_2_old)
        )

        obj = (
            objective(
                emp_cov,
                R,
                Z_0,
                Z_1,
                Z_2,
                W_0,
                W_1,
                W_2,
                alpha,
                tau,
                beta,
                eta,
                psi,
                phi,
            )
            if compute_objective
            else np.nan
        )

        check = Convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=np.sqrt(R.size + 4 * Z_1.size) * tol
            + rtol
            * max(
                np.sqrt(
                    squared_norm(R)
                    + squared_norm(Z_1)
                    + squared_norm(Z_2)
                    + squared_norm(W_1)
                    + squared_norm(W_2)
                ),
                np.sqrt(
                    squared_norm(Z_0 - W_0)
                    + squared_norm(Z_0[:-1])
                    + squared_norm(Z_0[1:])
                    + squared_norm(W_0[:-1])
                    + squared_norm(W_0[1:])
                ),
            ),
            e_dual=np.sqrt(R.size + 4 * Z_1.size) * tol
            + rtol
            * rho
            * (
                np.sqrt(
                    squared_norm(X_0)
                    + squared_norm(X_1)
                    + squared_norm(X_2)
                    + squared_norm(U_1)
                    + squared_norm(U_2)
                )
            ),
        )

        R_old = R.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()
        W_1_old = W_1.copy()
        W_2_old = W_2.copy()

        if verbose:
            print(check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        rho_new = update_rho(
            rho, rnorm, snorm, iteration=iteration_, **(update_rho_options or {})
        )
        # scaled dual variables should be also rescaled
        X_0 *= rho / rho_new
        X_1 *= rho / rho_new
        X_2 *= rho / rho_new
        U_1 *= rho / rho_new
        U_2 *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [Z_0, W_0]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class LatentTimeMatrixDecomposition(LatentTimeGraphicalLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Parameters
    ----------
    alpha : positive float, default 0.01
        Regularization parameter for precision matrix. The higher alpha,
        the more regularization, the sparser the inverse covariance.

    tau : positive float, default 1
        Regularization parameter for latent variables matrix. The higher tau,
        the more regularization, the lower rank of the latent matrix.

    beta : positive float, default 1
        Regularization parameter to constrain precision matrices in time.
        The higher beta, the more regularization,
        and consecutive precision matrices in time are more similar.

    psi : {'laplacian', 'l1', 'l2', 'linf', 'node'}, default 'laplacian'
        Type of norm to enforce for consecutive precision matrices in time.

    eta : positive float, default 1
        Regularization parameter to constrain latent matrices in time.
        The higher eta, the more regularization,
        and consecutive latent matrices in time are more similar.

    phi : {'laplacian', 'l1', 'l2', 'linf', 'node'}, default 'laplacian'
        Type of norm to enforce for consecutive latent matrices in time.

    rho : positive float, default 1
        Augmented Lagrangian parameter.

    over_relax : positive float, deafult 1
        Over-relaxation parameter (typically between 1.0 and 1.8).

    tol : positive float, default 1e-4
        Absolute tolerance to declare convergence.

    rtol : positive float, default 1e-4
        Relative tolerance to declare convergence.

    max_iter : integer, default 100
        The maximum number of iterations.

    verbose : boolean, default False
        If verbose is True, the objective function, rnorm and snorm are
        printed at each iteration.

    assume_centered : boolean, default False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    time_on_axis : {'first', 'last'}, default 'first'
        If data have time as the last dimension, set this to 'last'.
        Useful to use scikit-learn functions as train_test_split.

    update_rho_options : dict, default None
        Options for the update of rho. See `update_rho` function for details.

    compute_objective : boolean, default True
        Choose if compute the objective function during iterations
        (only useful if `verbose=True`).

    mode : {'admm'}, default 'admm'
        Minimisation algorithm. At the moment, only 'admm' is available,
        so this is ignored.

    Attributes
    ----------
    precision_ : array-like, shape (n_times, n_features, n_features)
        Estimated pseudo inverse matrix.

    latent_ : array-like, shape (n_times, n_features, n_features)
        Estimated latent variable matrix.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(
        self,
        alpha=0.01,
        tau=1.0,
        beta=1.0,
        eta=1.0,
        mode="admm",
        rho=1.0,
        time_on_axis="first",
        tol=1e-4,
        rtol=1e-4,
        psi="laplacian",
        phi="laplacian",
        max_iter=100,
        verbose=False,
        assume_centered=False,
        update_rho_options=None,
        compute_objective=True,
    ):
        super(LatentTimeMatrixDecomposition, self).__init__(
            alpha=alpha,
            beta=beta,
            tau=tau,
            eta=eta,
            mode=mode,
            rho=rho,
            tol=tol,
            rtol=rtol,
            psi=psi,
            phi=phi,
            max_iter=max_iter,
            verbose=verbose,
            assume_centered=assume_centered,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective,
        )
        self.time_on_axis = time_on_axis

    def _fit(self, X):
        """Fit the LatentTimeMatrixDecomposition model to X.

        Parameters
        ----------
        X : ndarray, shape (n_time, n_samples, n_features), or
                (n_samples, n_features, n_time)
            Matrix to decompose.

        """
        self.precision_, self.latent_, self.n_iter_ = latent_time_matrix_decomposition(
            X,
            alpha=self.alpha,
            tau=self.tau,
            rho=self.rho,
            beta=self.beta,
            eta=self.eta,
            mode=self.mode,
            tol=self.tol,
            rtol=self.rtol,
            psi=self.psi,
            phi=self.phi,
            max_iter=self.max_iter,
            verbose=self.verbose,
            return_n_iter=True,
            return_history=False,
            update_rho_options=self.update_rho_options,
            compute_objective=self.compute_objective,
        )
        self.reconstruction_err_ = squared_norm(X - self.get_observed_precision())
        return self

    def fit(self, X, y=None):
        """Fit the LatentTimeMatrixDecomposition model to X.

        Parameters
        ----------
        X : ndarray, shape (n_time, n_samples, n_features), or
                (n_samples, n_features, n_time)
            Matrix to decompose.
            If shape is (n_samples, n_features, n_time), then set
            `time_on_axis = 'last'`.
        y : (ignored)

        """
        X, _, _, _ = check_input(X, time_on_axis=self.time_on_axis, estimator=self)

        return self._fit(X)
