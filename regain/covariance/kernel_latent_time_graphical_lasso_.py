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
"""Graphical latent variable models selection over time via ADMM.

This adds the possibility to specify a temporal constraint with a kernel
function.
"""
from __future__ import division

import warnings
from functools import partial

import numpy as np
from scipy import linalg
from six.moves import map, range, zip
from sklearn.cluster import AgglomerativeClustering
from sklearn.gaussian_process import kernels
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted

from regain.covariance.kernel_time_graphical_lasso_ import KernelTimeGraphicalLasso, init_precision
from regain.covariance.kernel_time_graphical_lasso_ import objective as obj_ktgl
from regain.covariance.kernel_time_graphical_lasso_ import precision_similarity
from regain.prox import prox_logdet, prox_trace_indicator, soft_thresholding
from regain.update_rules import update_rho
from regain.utils import convergence
from regain.validation import check_norm_prox


def objective(S, n_samples, R, Z_0, Z_M, W_0, W_M, alpha, tau, kernel_psi, kernel_phi, psi, phi):
    """Objective function for latent variable time-varying graphical lasso."""
    obj = obj_ktgl(n_samples, S, R, Z_0, Z_M, alpha, kernel_psi, psi)
    if isinstance(tau, np.ndarray):
        obj += sum(np.linalg.norm(t * w, ord="nuc") for t, w in zip(tau, W_0))
    else:
        obj += tau * sum(map(partial(np.linalg.norm, ord="nuc"), W_0))

    for m in range(1, W_0.shape[0]):
        # all possible markovians jumps
        W_L, W_R = W_M[m]
        obj += np.sum(np.array(list(map(phi, W_R - W_L))) * np.diag(kernel_phi, m))

    return obj


def kernel_latent_time_graphical_lasso(
    emp_cov,
    alpha=0.01,
    tau=1.0,
    rho=1.0,
    kernel_psi=None,
    kernel_phi=None,
    max_iter=100,
    verbose=False,
    psi="laplacian",
    phi="laplacian",
    mode="admm",
    tol=1e-4,
    rtol=1e-4,
    assume_centered=False,
    n_samples=None,
    return_history=False,
    return_n_iter=True,
    update_rho_options=None,
    compute_objective=True,
    init="empirical",
):
    r"""Time-varying latent variable graphical lasso solver.

    Solves the following problem via ADMM:
        min sum_{i=1}^T -n_i log_likelihood(K_i-L_i) + alpha ||K_i||_{od,1}
            + tau ||L_i||_*
            + sum_{s>t}^T k_psi(s,t) Psi(K_s - K_t)
            + sum_{s>t}^T k_phi(s,t)(L_s - L_t)

    where S is the empirical covariance of the data
    matrix D (training observations by features).

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
    n_times, _, n_features = emp_cov.shape

    if kernel_psi is None:
        kernel_psi = np.eye(n_times)
    if kernel_phi is None:
        kernel_phi = np.eye(n_times)

    Z_0 = init_precision(emp_cov, mode=init)
    W_0 = np.zeros_like(Z_0)
    X_0 = np.zeros_like(Z_0)
    R_old = np.zeros_like(Z_0)

    Z_M, Z_M_old = {}, {}
    Y_M = {}
    W_M, W_M_old = {}, {}
    U_M = {}
    for m in range(1, n_times):
        Z_L = Z_0.copy()[:-m]
        Z_R = Z_0.copy()[m:]
        Z_M[m] = (Z_L, Z_R)

        W_L = np.zeros_like(Z_L)
        W_R = np.zeros_like(Z_R)
        W_M[m] = (W_L, W_R)

        Y_L = np.zeros_like(Z_L)
        Y_R = np.zeros_like(Z_R)
        Y_M[m] = (Y_L, Y_R)

        U_L = np.zeros_like(W_L)
        U_R = np.zeros_like(W_R)
        U_M[m] = (U_L, U_R)

        Z_L_old = np.zeros_like(Z_L)
        Z_R_old = np.zeros_like(Z_R)
        Z_M_old[m] = (Z_L_old, Z_R_old)

        W_L_old = np.zeros_like(W_L)
        W_R_old = np.zeros_like(W_R)
        W_M_old[m] = (W_L_old, W_R_old)

    if n_samples is None:
        n_samples = np.ones(n_times)

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
        for m in range(1, n_times):
            A[:-m] += Z_M[m][0] - Y_M[m][0]
            A[m:] += Z_M[m][1] - Y_M[m][1]

        A /= n_times
        Z_0 = soft_thresholding(A, lamda=alpha / (rho * n_times))

        # update W_0
        A = Z_0 - R - X_0
        for m in range(1, n_times):
            A[:-m] += W_M[m][0] - U_M[m][0]
            A[m:] += W_M[m][1] - U_M[m][1]

        A /= n_times
        A += A.transpose(0, 2, 1)
        A /= 2.0

        W_0 = np.array([prox_trace_indicator(a, lamda=tau / (rho * n_times)) for a in A])

        # update residuals
        X_0 += R - Z_0 + W_0

        for m in range(1, n_times):
            # other Zs
            Y_L, Y_R = Y_M[m]
            A_L = Z_0[:-m] + Y_L
            A_R = Z_0[m:] + Y_R
            if not psi_node_penalty:
                prox_e = prox_psi(A_R - A_L, lamda=2.0 * np.diag(kernel_psi, m)[:, None, None] / rho)
                Z_L = 0.5 * (A_L + A_R - prox_e)
                Z_R = 0.5 * (A_L + A_R + prox_e)
            else:
                Z_L, Z_R = prox_psi(
                    np.concatenate((A_L, A_R), axis=1),
                    lamda=0.5 * np.diag(kernel_psi, m)[:, None, None] / rho,
                    rho=rho,
                    tol=tol,
                    rtol=rtol,
                    max_iter=max_iter,
                )
            Z_M[m] = (Z_L, Z_R)

            # update other residuals
            Y_L += Z_0[:-m] - Z_L
            Y_R += Z_0[m:] - Z_R

            # other Ws
            U_L, U_R = U_M[m]
            A_L = W_0[:-m] + U_L
            A_R = W_0[m:] + U_R
            if not phi_node_penalty:
                prox_e = prox_phi(A_R - A_L, lamda=2.0 * np.diag(kernel_phi, m)[:, None, None] / rho)
                W_L = 0.5 * (A_L + A_R - prox_e)
                W_R = 0.5 * (A_L + A_R + prox_e)
            else:
                W_L, W_R = prox_phi(
                    np.concatenate((A_L, A_R), axis=1),
                    lamda=0.5 * np.diag(kernel_phi, m)[:, None, None] / rho,
                    rho=rho,
                    tol=tol,
                    rtol=rtol,
                    max_iter=max_iter,
                )
            W_M[m] = (W_L, W_R)

            # update other residuals
            U_L += W_0[:-m] - W_L
            U_R += W_0[m:] - W_R

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(R - Z_0 + W_0)
            + sum(
                squared_norm(Z_0[:-m] - Z_M[m][0])
                + squared_norm(Z_0[m:] - Z_M[m][1])
                + squared_norm(W_0[:-m] - W_M[m][0])
                + squared_norm(W_0[m:] - W_M[m][1])
                for m in range(1, n_times)
            )
        )

        snorm = rho * np.sqrt(
            squared_norm(R - R_old)
            + sum(
                squared_norm(Z_M[m][0] - Z_M_old[m][0])
                + squared_norm(Z_M[m][1] - Z_M_old[m][1])
                + squared_norm(W_M[m][0] - W_M_old[m][0])
                + squared_norm(W_M[m][1] - W_M_old[m][1])
                for m in range(1, n_times)
            )
        )

        obj = (
            objective(emp_cov, n_samples, R, Z_0, Z_M, W_0, W_M, alpha, tau, kernel_psi, kernel_phi, psi, phi)
            if compute_objective
            else np.nan
        )

        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=n_features * np.sqrt(n_times * (2 * n_times - 1)) * tol
            + rtol
            * max(
                np.sqrt(
                    squared_norm(R)
                    + sum(
                        squared_norm(Z_M[m][0])
                        + squared_norm(Z_M[m][1])
                        + squared_norm(W_M[m][0])
                        + squared_norm(W_M[m][1])
                        for m in range(1, n_times)
                    )
                ),
                np.sqrt(
                    squared_norm(Z_0 - W_0)
                    + sum(
                        squared_norm(Z_0[:-m]) + squared_norm(Z_0[m:]) + squared_norm(W_0[:-m]) + squared_norm(W_0[m:])
                        for m in range(1, n_times)
                    )
                ),
            ),
            e_dual=n_features * np.sqrt(n_times * (2 * n_times - 1)) * tol
            + rtol
            * rho
            * np.sqrt(
                squared_norm(X_0)
                + sum(
                    squared_norm(Y_M[m][0])
                    + squared_norm(Y_M[m][1])
                    + squared_norm(U_M[m][0])
                    + squared_norm(U_M[m][1])
                    for m in range(1, n_times)
                )
            ),
        )

        R_old = R.copy()
        for m in range(1, n_times):
            Z_M_old[m] = (Z_M[m][0].copy(), Z_M[m][1].copy())
            W_M_old[m] = (W_M[m][0].copy(), W_M[m][1].copy())

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f," "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_, **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        X_0 *= rho / rho_new
        for m in range(1, n_times):
            Y_L, Y_R = Y_M[m]
            Y_L *= rho / rho_new
            Y_R *= rho / rho_new

            U_L, U_R = U_M[m]
            U_L *= rho / rho_new
            U_R *= rho / rho_new
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


class KernelLatentTimeGraphicalLasso(KernelTimeGraphicalLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Parameters
    ----------
    alpha : positive float, default 0.01
        Regularization parameter for precision matrix. The higher alpha,
        the more regularization, the sparser the inverse covariance.

    tau : positive float, default 1
        Regularization parameter for latent variables matrix. The higher tau,
        the more regularization, the lower rank of the latent matrix.

    kernel_{psi,phi} : ndarray, default None
        Normalised temporal kernel (1 on the diagonal),
        with dimensions equal to the dimensionality of the data set.
        If None, it is interpreted as an identity matrix, where there is no
        constraint on the temporal behaviour of {precision,latent} matrices.

    {psi,phi} : {'laplacian', 'l1', 'l2', 'linf', 'node'}, default 'laplacian'
        Type of norm to enforce for consecutive {precision,latent} matrices
        in time.

    rho : positive float, default 1
        Augmented Lagrangian parameter.

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
        kernel_psi=None,
        kernel_phi=None,
        rho=1.0,
        tol=1e-4,
        rtol=1e-4,
        psi="laplacian",
        phi="laplacian",
        max_iter=100,
        verbose=False,
        assume_centered=False,
        return_history=False,
        update_rho_options=None,
        compute_objective=True,
        ker_psi_param=1,
        ker_phi_param=1,
        init="empirical",
    ):
        super(KernelLatentTimeGraphicalLasso, self).__init__(
            alpha=alpha,
            rho=rho,
            tol=tol,
            rtol=rtol,
            max_iter=max_iter,
            verbose=verbose,
            assume_centered=assume_centered,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective,
            return_history=return_history,
            psi=psi,
            init=init,
        )
        self.kernel_psi = kernel_psi
        self.kernel_phi = kernel_phi
        self.tau = tau
        self.phi = phi
        self.ker_psi_param = ker_psi_param
        self.ker_phi_param = ker_phi_param

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
        # TODO auto discover parameter
        if callable(self.kernel_phi):
            try:
                # this works if it is a ExpSineSquared or RBF kernel
                kernel_phi = self.kernel_phi(length_scale=self.ker_phi_param)(self.classes_[:, None])
            except TypeError:
                # maybe it's a ConstantKernel
                kernel_phi = self.kernel_phi(constant_value=self.ker_phi_param)(self.classes_[:, None])

        else:
            kernel_phi = self.kernel_phi
            if kernel_phi.shape[0] != self.classes_.size:
                raise ValueError(
                    "kernel_phi size does not match classes of samples, "
                    "got {} classes and kernel_phi has shape {}".format(self.classes_.size, kernel_phi.shape[0])
                )
        if callable(self.kernel_psi):
            try:
                # this works if it is a ExpSineSquared kernel
                kernel_psi = self.kernel_psi(length_scale=self.ker_psi_param)(self.classes_[:, None])
            except TypeError:
                # maybe it's a ConstantKernel
                kernel_psi = self.kernel_psi(constant_value=self.ker_psi_param)(self.classes_[:, None])
        else:
            kernel_psi = self.kernel_psi
            if kernel_psi.shape[0] != self.classes_.size:
                raise ValueError(
                    "kernel_psi size does not match classes of samples, "
                    "got {} classes and kernel_psi has shape {}".format(self.classes_.size, kernel_psi.shape[0])
                )

        out = kernel_latent_time_graphical_lasso(
            emp_cov,
            alpha=self.alpha,
            tau=self.tau,
            rho=self.rho,
            kernel_phi=kernel_phi,
            kernel_psi=kernel_psi,
            n_samples=n_samples,
            tol=self.tol,
            rtol=self.rtol,
            psi=self.psi,
            max_iter=self.max_iter,
            verbose=self.verbose,
            return_n_iter=True,
            return_history=self.return_history,
            update_rho_options=self.update_rho_options,
            compute_objective=self.compute_objective,
            init=self.init,
        )
        if self.return_history:
            self.precision_, self.latent_, self.covariance_, self.history_, self.n_iter_ = out
        else:
            self.precision_, self.latent_, self.covariance_, self.n_iter_ = out

        return self


class SimilarityLatentTimeGraphicalLasso(KernelLatentTimeGraphicalLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Parameters
    ----------
    alpha : positive float, default 0.01
        Regularization parameter for precision matrix. The higher alpha,
        the more regularization, the sparser the inverse covariance.

    tau : positive float, default 1
        Regularization parameter for latent variables matrix. The higher tau,
        the more regularization, the lower rank of the latent matrix.

    kernel_{psi,phi} : ndarray, default None
        Normalised temporal kernel (1 on the diagonal),
        with dimensions equal to the dimensionality of the data set.
        If None, it is interpreted as an identity matrix, where there is no
        constraint on the temporal behaviour of {precision,latent} matrices.

    {psi,phi} : {'laplacian', 'l1', 'l2', 'linf', 'node'}, default 'laplacian'
        Type of norm to enforce for consecutive {precision,latent} matrices
        in time.

    rho : positive float, default 1
        Augmented Lagrangian parameter.

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
        beta=1.0,
        tau=1.0,
        eta=1.0,
        kernel_psi=None,
        kernel_phi=None,
        rho=1.0,
        tol=1e-4,
        rtol=1e-4,
        psi="laplacian",
        phi="laplacian",
        max_iter=100,
        verbose=False,
        assume_centered=False,
        return_history=False,
        update_rho_options=None,
        compute_objective=True,
        ker_psi_param=1,
        ker_phi_param=1,
        max_iter_ext=100,
        init="empirical",
        eps=1e-6,
        n_clusters=None,
    ):
        super(SimilarityLatentTimeGraphicalLasso, self).__init__(
            alpha=alpha,
            tau=tau,
            phi=phi,
            psi=psi,
            rho=rho,
            tol=tol,
            rtol=rtol,
            max_iter=max_iter,
            verbose=verbose,
            assume_centered=assume_centered,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective,
            return_history=return_history,
            kernel_psi=kernel_psi,
            kernel_phi=kernel_phi,
            ker_psi_param=ker_psi_param,
            ker_phi_param=ker_phi_param,
            init=init,
        )
        self.beta = beta
        self.eta = eta
        self.max_iter_ext = max_iter_ext
        self.eps = eps
        self.n_clusters = n_clusters

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
        if self.kernel_psi is None:
            n_times = emp_cov.shape[0]

            if self.kernel_phi is None or callable(self.kernel_phi):
                # raise ValueError('not implemented')
                # mimic LTGL
                kernel_phi = np.eye(n_times)
                np.fill_diagonal(kernel_phi[:, 1:], self.eta)
                np.fill_diagonal(kernel_phi[1:], self.eta)

            # discover best kernel parameter via EM
            # initialise precision matrices, as warm start
            self.precision_ = init_precision(emp_cov, mode=self.init)
            self.latent_ = np.zeros_like(self.precision_)
            theta_old = np.zeros(n_times * (n_times - 1) // 2)
            kernel_psi = np.eye(n_times)

            psi, _, _ = check_norm_prox(self.psi)
            if self.n_clusters is None:
                self.n_clusters = n_times

            for i in range(self.max_iter_ext):
                # E step - discover best kernel
                theta = precision_similarity(self.get_precision(), psi)

                # if i > 0 and np.linalg.norm(theta_old -
                #                             theta) / theta.size < self.eps:
                #     break

                # kernel_psi = theta * self.beta
                kernel_psi = theta
                labels_pred = AgglomerativeClustering(
                    n_clusters=self.n_clusters, affinity="precomputed", linkage="complete"
                ).fit_predict(kernel_psi)
                if i > 0 and np.linalg.norm(labels_pred - labels_pred_old) / labels_pred.size < self.eps:
                    break
                kernel_psi = kernels.RBF(0.0001)(labels_pred[:, None]) + kernels.RBF(self.beta)(
                    np.arange(n_times)[:, None]
                )

                # M step - fix the kernel matrix
                out = kernel_latent_time_graphical_lasso(
                    emp_cov,
                    alpha=self.alpha,
                    tau=self.tau,
                    rho=self.rho,
                    kernel_phi=self.kernel_phi,
                    kernel_psi=kernel_psi,
                    n_samples=n_samples,
                    tol=self.tol,
                    rtol=self.rtol,
                    psi=self.psi,
                    max_iter=self.max_iter,
                    verbose=self.verbose,
                    return_n_iter=True,
                    return_history=self.return_history,
                    update_rho_options=self.update_rho_options,
                    compute_objective=self.compute_objective,
                    init=self.precision_,
                )

                if self.return_history:
                    (self.precision_, self.latent_, self.covariance_, self.history_, self.n_iter_) = out
                else:
                    (self.precision_, self.latent_, self.covariance_, self.n_iter_) = out
                theta_old = theta
                labels_pred_old = labels_pred
            else:
                warnings.warn("theta did not converge.")
            self.similarity_matrix_ = kernel_psi
        else:
            if callable(self.kernel_phi):
                try:
                    # this works if it is a ExpSineSquared or RBF kernel
                    kernel_phi = self.kernel_phi(length_scale=self.ker_phi_param)(self.classes_[:, None])
                except TypeError:
                    # maybe it's a ConstantKernel
                    kernel_phi = self.kernel_phi(constant_value=self.ker_phi_param)(self.classes_[:, None])

            else:
                kernel_phi = self.kernel_phi
                if kernel_phi.shape[0] != self.classes_.size:
                    raise ValueError(
                        "kernel_phi size does not match classes of samples, "
                        "got {} classes and kernel_phi has shape {}".format(self.classes_.size, kernel_phi.shape[0])
                    )
            if callable(self.kernel_psi):
                try:
                    # this works if it is a ExpSineSquared kernel
                    kernel_psi = self.kernel_psi(length_scale=self.ker_psi_param)(self.classes_[:, None])
                except TypeError:
                    # maybe it's a ConstantKernel
                    kernel_psi = self.kernel_psi(constant_value=self.ker_psi_param)(self.classes_[:, None])
            else:
                kernel_psi = self.kernel_psi
                if kernel_psi.shape[0] != self.classes_.size:
                    raise ValueError(
                        "kernel_psi size does not match classes of samples, "
                        "got {} classes and kernel_psi has shape {}".format(self.classes_.size, kernel_psi.shape[0])
                    )

            out = kernel_latent_time_graphical_lasso(
                emp_cov,
                alpha=self.alpha,
                tau=self.tau,
                rho=self.rho,
                kernel_phi=kernel_phi,
                kernel_psi=kernel_psi,
                n_samples=n_samples,
                tol=self.tol,
                rtol=self.rtol,
                psi=self.psi,
                max_iter=self.max_iter,
                verbose=self.verbose,
                return_n_iter=True,
                return_history=self.return_history,
                update_rho_options=self.update_rho_options,
                compute_objective=self.compute_objective,
                init=self.init,
            )
            if self.return_history:
                (self.precision_, self.latent_, self.covariance_, self.history_, self.n_iter_) = out
            else:
                (self.precision_, self.latent_, self.covariance_, self.n_iter_) = out

        return self

    def transform(self, X, y=None):
        """Possibility to add in a Pipeline."""
        check_is_fitted(self, ["similarity_matrix_"])

        return self.similarity_matrix_
