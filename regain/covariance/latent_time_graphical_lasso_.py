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
"""Graphical latent variable models selection over time via ADMM."""
from __future__ import division

import warnings
from functools import partial

import numpy as np
from scipy import linalg
from six.moves import map, range, zip
from sklearn.utils.extmath import squared_norm

from regain.covariance.time_graphical_lasso_ import TimeGraphicalLasso, init_precision
from regain.covariance.time_graphical_lasso_ import objective as obj_tgl
from regain.prox import prox_logdet, prox_trace_indicator, soft_thresholding
from regain.update_rules import update_rho
from regain.utils import convergence
from regain.validation import check_norm_prox


def objective(S, n_samples, R, Z_0, Z_1, Z_2, W_0, W_1, W_2, alpha, tau, beta, eta, psi, phi):
    """Objective function for latent variable time-varying graphical lasso."""
    # obj = sum(- n * logl(s, r) for s, r, n in zip(S, R, n_samples))
    obj = obj_tgl(n_samples, S, R, Z_0, Z_1, Z_2, alpha, beta, psi)

    if isinstance(tau, np.ndarray):
        obj += sum(np.linalg.norm(t * w, ord="nuc") for t, w in zip(tau, W_0))
    else:
        obj += tau * sum(map(partial(np.linalg.norm, ord="nuc"), W_0))

    if isinstance(eta, np.ndarray):
        obj += sum(b[0][0] * m for b, m in zip(eta, map(phi, W_2 - W_1)))
    else:
        obj += eta * sum(map(phi, W_2 - W_1))
    return obj


def latent_time_graphical_lasso(
    emp_cov,
    alpha=0.01,
    tau=1.0,
    rho=1.0,
    beta=1.0,
    eta=1.0,
    max_iter=100,
    n_samples=None,
    verbose=False,
    psi="laplacian",
    phi="laplacian",
    mode="admm",
    tol=1e-4,
    rtol=1e-4,
    return_history=False,
    return_n_iter=True,
    update_rho_options=None,
    compute_objective=True,
    init="empirical",
):
    r"""Latent variable time-varying graphical lasso solver.

    Solves the following problem via ADMM:
      min sum_{i=1}^T -n_i log_likelihood(S_i, K_i-L_i) + alpha ||K_i||_{od,1}
          + tau ||L_i||_*
          + beta sum_{i=2}^T Psi(K_i - K_{i-1})
          + eta sum_{i=2}^T Phi(L_i - L_{i-1})

    where S_i = (1/n_i) X_i^T \times X_i is the empirical covariance of data
    matrix X (training observations by features).

    Parameters
    ----------
    emp_cov : ndarray, shape (n_features, n_features)
        Empirical covariance of data.
    alpha, tau, beta, eta : float, optional
        Regularisation parameters.
    rho : float, optional
        Augmented Lagrangian parameter.
    max_iter : int, optional
        Maximum number of iterations.
    n_samples : ndarray
        Number of samples available for each time point.
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
    update_rho_options : dict, optional
        Arguments for the rho update.
        See regain.update_rules.update_rho function for more information.
    compute_objective : bool, default True
        Choose to compute the objective value.
    init : {'empirical', 'zeros', ndarray}, default 'empirical'
        How to initialise the inverse covariance matrix. Default is take
        the empirical covariance and inverting it.

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

    Z_0 = init_precision(emp_cov, mode=init)
    Z_1 = Z_0.copy()[:-1]
    Z_2 = Z_0.copy()[1:]
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

    if n_samples is None:
        n_samples = np.ones(emp_cov.shape[0])

    checks = []
    for iteration_ in range(max_iter):
        # update R
        A = Z_0 - W_0 - X_0
        A += A.transpose(0, 2, 1)
        A /= 2.0
        A *= -rho / n_samples[:, None, None]
        A += emp_cov
        # A = emp_cov / rho - A

        R = np.array([prox_logdet(a, lamda=ni / rho) for a, ni in zip(A, n_samples)])

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

        W_0 = np.array([prox_trace_indicator(a, lamda=tau / (rho * div)) for a, div in zip(A, divisor)])

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
            objective(emp_cov, n_samples, R, Z_0, Z_1, Z_2, W_0, W_1, W_2, alpha, tau, beta, eta, psi, phi)
            if compute_objective
            else np.nan
        )

        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=np.sqrt(R.size + 4 * Z_1.size) * tol
            + rtol
            * max(
                np.sqrt(
                    squared_norm(R) + squared_norm(Z_1) + squared_norm(Z_2) + squared_norm(W_1) + squared_norm(W_2)
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
                    squared_norm(X_0) + squared_norm(X_1) + squared_norm(X_2) + squared_norm(U_1) + squared_norm(U_2)
                )
            ),
        )

        R_old = R.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()
        W_1_old = W_1.copy()
        W_2_old = W_2.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f," "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_, **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        X_0 *= rho / rho_new
        X_1 *= rho / rho_new
        X_2 *= rho / rho_new
        U_1 *= rho / rho_new
        U_2 *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    covariance_ = np.array([linalg.pinvh(x) for x in Z_0])
    return_list = [Z_0, W_0, covariance_]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class LatentTimeGraphicalLasso(TimeGraphicalLasso):
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

    init : {'empirical', 'zeros', ndarray}, default 'empirical'
        How to initialise the inverse covariance matrix. Default is take
        the empirical covariance and inverting it.

    Attributes
    ----------
    covariance_ : array-like, shape (n_times, n_features, n_features)
        Estimated covariance matrix

    precision_, latent_ : array-like, shape (n_times, n_features, n_features)
        Estimated precision and latent variables matrix.

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
        tol=1e-4,
        rtol=1e-4,
        psi="laplacian",
        phi="laplacian",
        max_iter=100,
        verbose=False,
        assume_centered=False,
        update_rho_options=None,
        compute_objective=True,
        init="empirical",
    ):
        super(LatentTimeGraphicalLasso, self).__init__(
            alpha=alpha,
            beta=beta,
            mode=mode,
            rho=rho,
            tol=tol,
            rtol=rtol,
            psi=psi,
            max_iter=max_iter,
            verbose=verbose,
            assume_centered=assume_centered,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective,
            init=init,
        )
        self.tau = tau
        self.eta = eta
        self.phi = phi

    def get_observed_precision(self):
        """Getter for the observed precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.
            Note that this is the observed precision matrix.

        """
        return self.precision_ - self.latent_

    def _fit(self, emp_cov, n_samples):
        """Fit the LatentTimeGraphicalLasso model to X.

        Parameters
        ----------
        emp_cov : ndarray, shape (n_features, n_features)
            Empirical covariance of data.

        """
        self.precision_, self.latent_, self.covariance_, self.n_iter_ = latent_time_graphical_lasso(
            emp_cov,
            n_samples=n_samples,
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
            init=self.init,
        )
        return self
