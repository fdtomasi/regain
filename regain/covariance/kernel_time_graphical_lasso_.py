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
"""Sparse inverse covariance selection over time via ADMM.

This adds the possibility to specify a temporal constraint with a kernel
function.
"""
from __future__ import division

import warnings

import numpy as np
from scipy import linalg
from six.moves import map, range, zip
from sklearn.cluster import AgglomerativeClustering
from sklearn.gaussian_process import kernels
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted

from regain.covariance.time_graphical_lasso_ import TimeGraphicalLasso, init_precision, loss
from regain.norm import l1_od_norm
from regain.prox import prox_logdet, soft_thresholding
from regain.update_rules import update_rho
from regain.utils import convergence
from regain.validation import check_norm_prox

# from regain.clustering import graph_k_means


def objective(n_samples, S, K, Z_0, Z_M, alpha, kernel, psi):
    """Objective function for time-varying graphical lasso."""
    obj = loss(S, K, n_samples=n_samples)
    if isinstance(alpha, np.ndarray):
        obj += sum(l1_od_norm(a * z) for a, z in zip(alpha, Z_0))
    else:
        obj += alpha * sum(map(l1_od_norm, Z_0))

    for m in range(1, Z_0.shape[0]):
        # all possible markovians jumps
        Z_L, Z_R = Z_M[m]
        obj += np.sum(np.array(list(map(psi, Z_R - Z_L))) * np.diag(kernel, m))

    return obj


def kernel_time_graphical_lasso(
    emp_cov,
    alpha=0.01,
    rho=1,
    kernel=None,
    max_iter=100,
    n_samples=None,
    verbose=False,
    psi="laplacian",
    tol=1e-4,
    rtol=1e-4,
    return_history=False,
    return_n_iter=True,
    mode="admm",
    update_rho_options=None,
    compute_objective=True,
    stop_at=None,
    stop_when=1e-4,
    init="empirical",
):
    """Time-varying graphical lasso solver.

    Solves the following problem via ADMM:
        min sum_{i=1}^T -n_i log_likelihood(K_i-L_i) + alpha ||K_i||_{od,1}
            + sum_{s>t}^T k_psi(s,t) Psi(K_s - K_t)

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    emp_cov : ndarray, shape (n_features, n_features)
        Empirical covariance of data.
    alpha, beta : float, optional
        Regularisation parameter.
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
    init : {'empirical', 'zeros', ndarray}, default 'empirical'
        How to initialise the inverse covariance matrix. Default is take
        the empirical covariance and inverting it.

    Returns
    -------
    X : numpy.array, 2-dimensional
        Solution to the problem.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    psi, prox_psi, psi_node_penalty = check_norm_prox(psi)
    n_times, _, n_features = emp_cov.shape

    if kernel is None:
        kernel = np.eye(n_times)

    Z_0 = init_precision(emp_cov, mode=init)
    U_0 = np.zeros_like(Z_0)
    Z_0_old = np.zeros_like(Z_0)

    Z_M, Z_M_old = {}, {}
    U_M = {}
    for m in range(1, n_times):
        # all possible markovians jumps
        Z_L = Z_0.copy()[:-m]
        Z_R = Z_0.copy()[m:]
        Z_M[m] = (Z_L, Z_R)

        U_L = np.zeros_like(Z_L)
        U_R = np.zeros_like(Z_R)
        U_M[m] = (U_L, U_R)

        Z_L_old = np.zeros_like(Z_L)
        Z_R_old = np.zeros_like(Z_R)
        Z_M_old[m] = (Z_L_old, Z_R_old)

    if n_samples is None:
        n_samples = np.ones(n_times)

    checks = [convergence(obj=objective(n_samples, emp_cov, Z_0, Z_0, Z_M, alpha, kernel, psi))]
    for iteration_ in range(max_iter):
        # update K
        A = Z_0 - U_0
        for m in range(1, n_times):
            A[:-m] += Z_M[m][0] - U_M[m][0]
            A[m:] += Z_M[m][1] - U_M[m][1]

        A /= n_times
        # soft_thresholding_ = partial(soft_thresholding, lamda=alpha / rho)
        # K = np.array(map(soft_thresholding_, A))
        A += A.transpose(0, 2, 1)
        A /= 2.0

        A *= -rho * n_times / n_samples[:, None, None]
        A += emp_cov

        K = np.array([prox_logdet(a, lamda=ni / (rho * n_times)) for a, ni in zip(A, n_samples)])

        # update Z_0
        A = K + U_0
        A += A.transpose(0, 2, 1)
        A /= 2.0
        Z_0 = soft_thresholding(A, lamda=alpha / rho)

        # update residuals
        U_0 += K - Z_0

        # other Zs
        for m in range(1, n_times):
            U_L, U_R = U_M[m]
            A_L = K[:-m] + U_L
            A_R = K[m:] + U_R
            if not psi_node_penalty:
                prox_e = prox_psi(A_R - A_L, lamda=2.0 * np.diag(kernel, m)[:, None, None] / rho)
                Z_L = 0.5 * (A_L + A_R - prox_e)
                Z_R = 0.5 * (A_L + A_R + prox_e)
            else:
                Z_L, Z_R = prox_psi(
                    np.concatenate((A_L, A_R), axis=1),
                    lamda=0.5 * np.diag(kernel, m)[:, None, None] / rho,
                    rho=rho,
                    tol=tol,
                    rtol=rtol,
                    max_iter=max_iter,
                )
            Z_M[m] = (Z_L, Z_R)

            # update other residuals
            U_L += K[:-m] - Z_L
            U_R += K[m:] - Z_R

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(K - Z_0)
            + sum(squared_norm(K[:-m] - Z_M[m][0]) + squared_norm(K[m:] - Z_M[m][1]) for m in range(1, n_times))
        )

        snorm = rho * np.sqrt(
            squared_norm(Z_0 - Z_0_old)
            + sum(
                squared_norm(Z_M[m][0] - Z_M_old[m][0]) + squared_norm(Z_M[m][1] - Z_M_old[m][1])
                for m in range(1, n_times)
            )
        )

        obj = objective(n_samples, emp_cov, Z_0, K, Z_M, alpha, kernel, psi) if compute_objective else np.nan

        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=n_features * n_times * tol
            + rtol
            * max(
                np.sqrt(
                    squared_norm(Z_0)
                    + sum(squared_norm(Z_M[m][0]) + squared_norm(Z_M[m][1]) for m in range(1, n_times))
                ),
                np.sqrt(squared_norm(K) + sum(squared_norm(K[:-m]) + squared_norm(K[m:]) for m in range(1, n_times))),
            ),
            e_dual=n_features * n_times * tol
            + rtol
            * rho
            * np.sqrt(
                squared_norm(U_0) + sum(squared_norm(U_M[m][0]) + squared_norm(U_M[m][1]) for m in range(1, n_times))
            ),
        )
        Z_0_old = Z_0.copy()
        for m in range(1, n_times):
            Z_M_old[m] = (Z_M[m][0].copy(), Z_M[m][1].copy())

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f," "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        checks.append(check)
        if stop_at is not None:
            if abs(check.obj - stop_at) / abs(stop_at) < stop_when:
                break

        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_, **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        U_0 *= rho / rho_new
        for m in range(1, n_times):
            U_L, U_R = U_M[m]
            U_L *= rho / rho_new
            U_R *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    covariance_ = np.array([linalg.pinvh(x) for x in Z_0])
    return_list = [Z_0, covariance_]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_ + 1)
    return return_list


def objective_kernel(theta, K, psi, kernel, times):
    psi, _, _ = check_norm_prox(psi)
    try:
        # this works if it is a ExpSineSquared or RBF kernel
        kernel = kernel(length_scale=theta)(times)
    except TypeError:
        # maybe it's a ConstantKernel
        kernel = kernel(constant_value=theta)(times)

    obj = 0
    for m in range(1, K.shape[0]):
        # all possible markovians jumps
        obj += np.sum(np.array(list(map(psi, K[m:] - K[:-m]))) * np.diag(kernel, m))

    return obj


def objective_similarity(theta, K, times, psi):
    obj = 0
    n_times = K.shape[0]
    kernel = np.eye(n_times)
    idx = np.triu_indices(n_times, 1)
    kernel[idx] = theta
    kernel[idx[::-1]] = theta
    for m in range(1, n_times):
        # all possible markovians jumps
        obj += np.sum(np.array(list(map(psi, K[m:] - K[:-m]))) * np.diag(kernel, m))

    return obj


def precision_similarity(K, psi):
    n_times = K.shape[0]
    kernel = np.zeros((n_times, n_times))
    for m in range(1, n_times):
        # all possible markovians jumps
        dist = list(map(psi, K[m:] - K[:-m]))
        np.fill_diagonal(kernel[m:], dist)
        np.fill_diagonal(kernel[:, m:], dist)

    kernel -= np.min(kernel)
    kernel /= np.max(kernel)
    # kernel *= -1
    # kernel += 1
    return 1 - kernel  # 1. / (1 + kernel)


class KernelTimeGraphicalLasso(TimeGraphicalLasso):
    """Structure inference in time driven by a temporal kernel.

    Parameters
    ----------
    alpha : positive float, default 0.01
        Regularization parameter for precision matrix. The higher alpha,
        the more regularization, the sparser the inverse covariance.

    kernel : ndarray, default None
        Normalised temporal kernel (1 on the diagonal),
        with dimensions equal to the dimensionality of the data set.
        If None, it is interpreted as an identity matrix, where there is no
        constraint on the temporal behaviour of the precision matrices.

    psi : {'laplacian', 'l1', 'l2', 'linf', 'node'}, default 'laplacian'
        Type of norm to enforce for consecutive precision matrices in time.

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

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(
        self,
        alpha=0.01,
        beta=1,
        kernel=None,
        rho=1.0,
        tol=1e-4,
        rtol=1e-4,
        psi="laplacian",
        max_iter=100,
        verbose=False,
        assume_centered=False,
        return_history=False,
        update_rho_options=None,
        compute_objective=True,
        ker_param=1,
        max_iter_ext=100,
        init="empirical",
    ):
        super(KernelTimeGraphicalLasso, self).__init__(
            alpha=alpha,
            beta=beta,
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
        self.kernel = kernel
        self.ker_param = ker_param
        self.max_iter_ext = max_iter_ext

    def _fit(self, emp_cov, n_samples):
        if self.ker_param == "auto":
            from scipy.optimize import minimize_scalar

            if not callable(self.kernel):
                raise ValueError("kernel should be a function if ker_param=='auto'")
            # discover best kernel parameter via EM
            # initialise precision matrices, as warm start
            self.precision_ = init_precision(emp_cov, mode=self.init)
            theta_old = 0
            for i in range(self.max_iter_ext):
                # E step - discover best kernel parameter
                theta = minimize_scalar(
                    objective_kernel,
                    args=(self.precision_, self.psi, self.kernel, self.classes_[:, None]),
                    bounds=(0, emp_cov.shape[0]),
                    method="bounded",
                ).x

                if i > 0 and abs(theta_old - theta) < 1e-5:
                    break
                else:
                    print("Find new theta: %f" % theta)

                # M step
                try:
                    # this works if it is a ExpSineSquared or RBF kernel
                    kernel = self.kernel(length_scale=theta)(self.classes_[:, None])
                except TypeError:
                    # maybe it's a ConstantKernel
                    kernel = self.kernel(constant_value=theta)(self.classes_[:, None])

                out = kernel_time_graphical_lasso(
                    emp_cov,
                    alpha=self.alpha,
                    rho=self.rho,
                    kernel=kernel,
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
                    (self.precision_, self.covariance_, self.history_, self.n_iter_) = out
                else:
                    self.precision_, self.covariance_, self.n_iter_ = out
                theta_old = theta
            else:
                print("warning: theta not converged")

        else:
            if callable(self.kernel):
                try:
                    # this works if it is a ExpSineSquared or RBF kernel
                    kernel = self.kernel(length_scale=self.ker_param)(self.classes_[:, None])
                except TypeError:
                    # maybe it's a ConstantKernel
                    kernel = self.kernel(constant_value=self.ker_param)(self.classes_[:, None])
            else:
                kernel = self.kernel
                if kernel.shape[0] != self.classes_.size:
                    raise ValueError(
                        "Kernel size does not match classes of samples, "
                        "got {} classes and kernel has shape {}".format(self.classes_.size, kernel.shape[0])
                    )

            out = kernel_time_graphical_lasso(
                emp_cov,
                alpha=self.alpha,
                rho=self.rho,
                kernel=kernel,
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
                (self.precision_, self.covariance_, self.history_, self.n_iter_) = out
            else:
                self.precision_, self.covariance_, self.n_iter_ = out

        return self


class SimilarityTimeGraphicalLasso(KernelTimeGraphicalLasso):
    """Learn how to relate different precision matrices across times.

    Parameters
    ----------
    alpha : positive float, default 0.01
        Regularization parameter for precision matrix. The higher alpha,
        the more regularization, the sparser the inverse covariance.

    kernel : ndarray, default None
        Normalised temporal kernel (1 on the diagonal),
        with dimensions equal to the dimensionality of the data set.
        If None, it is interpreted as an identity matrix, where there is no
        constraint on the temporal behaviour of the precision matrices.

    psi : {'laplacian', 'l1', 'l2', 'linf', 'node'}, default 'laplacian'
        Type of norm to enforce for consecutive precision matrices in time.

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

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(
        self,
        alpha=0.01,
        beta=1,
        kernel=None,
        rho=1.0,
        tol=1e-4,
        rtol=1e-4,
        psi="laplacian",
        max_iter=100,
        verbose=False,
        assume_centered=False,
        return_history=False,
        update_rho_options=None,
        compute_objective=True,
        ker_param=1,
        max_iter_ext=100,
        init="empirical",
        eps=1e-6,
        n_clusters=None,
    ):
        super(SimilarityTimeGraphicalLasso, self).__init__(
            alpha=alpha,
            beta=beta,
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
        # in this class, `kernel` is either a matrix TxT or None
        # if None, automatically learn all the weights
        self.kernel = kernel
        self.max_iter_ext = max_iter_ext
        self.eps = eps
        self.n_clusters = n_clusters

    def _fit(self, emp_cov, n_samples):
        if self.kernel is None:
            # from scipy.optimize import minimize
            # discover best kernel parameter via EM
            # initialise precision matrices, as warm start
            self.precision_ = init_precision(emp_cov, mode=self.init)
            n_times = self.precision_.shape[0]
            theta_old = np.zeros(n_times * (n_times - 1) // 2)
            # idx = np.triu_indices(n_times, 1)
            kernel = np.eye(n_times)

            psi, _, _ = check_norm_prox(self.psi)
            if self.n_clusters is None:
                self.n_clusters = n_times

            for i in range(self.max_iter_ext):
                # E step - discover best kernel
                # , method='bounded'bounds=[(0, None)]*theta_old.size
                # theta = minimize(
                #     objective_similarity, theta_old,
                #     args=(self.precision_, self.classes_[:, None], psi)
                #     ).x
                # theta -= np.min(theta)
                # theta /= np.max(theta)
                theta = precision_similarity(self.precision_, psi)

                # if i > 0 and np.linalg.norm(theta_old -
                #                             theta) / theta.size < self.eps:
                #     break

                # kernel[idx] = theta
                # kernel[idx[::-1]] = theta
                kernel = theta

                labels_pred = AgglomerativeClustering(
                    n_clusters=self.n_clusters, affinity="precomputed", linkage="complete"
                ).fit_predict(kernel)
                if i > 0 and np.linalg.norm(labels_pred - labels_pred_old) / labels_pred.size < self.eps:
                    break
                kernel = kernels.RBF(0.0001)(labels_pred[:, None]) + kernels.RBF(self.beta)(
                    np.arange(n_times)[:, None]
                )

                # normalize_matrix(kernel_sum)
                # kernel += kerne * self.beta

                # M step - fix the kernel matrix
                out = kernel_time_graphical_lasso(
                    emp_cov,
                    alpha=self.alpha,
                    rho=self.rho,
                    kernel=kernel,
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
                    (self.precision_, self.covariance_, self.history_, self.n_iter_) = out
                else:
                    self.precision_, self.covariance_, self.n_iter_ = out
                theta_old = theta
                labels_pred_old = labels_pred
                # kernel = graph_k_means(
                #   list(self.precision_), 3, max_iter=100)
                # self.similarity_matrix = kernel
                # theta_old = kernel
                # if i > 0 and np.linalg.norm(theta_old -
                #                             kernel) / kernel.size < self.eps:
                #     break
            else:
                warnings.warn("theta did not converge.")
            self.similarity_matrix_ = kernel

        else:
            kernel = self.kernel
            if kernel.shape[0] != self.classes_.size:
                raise ValueError(
                    "Kernel size does not match classes of samples, "
                    "got {} classes and kernel has shape {}".format(self.classes_.size, kernel.shape[0])
                )

            out = kernel_time_graphical_lasso(
                emp_cov,
                alpha=self.alpha,
                rho=self.rho,
                kernel=kernel,
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
                (self.precision_, self.covariance_, self.history_, self.n_iter_) = out
            else:
                self.precision_, self.covariance_, self.n_iter_ = out

        return self

    def transform(self, X, y=None):
        """Possibility to add in a sklearn Pipeline."""
        check_is_fitted(self, ["similarity_matrix_"])

        return self.similarity_matrix_
