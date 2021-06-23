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

import warnings

import numpy as np

from six.moves import map, range, zip

from sklearn.base import BaseEstimator
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_X_y
from sklearn.cluster import AgglomerativeClustering
from sklearn.gaussian_process import kernels

from sklearn.utils.validation import check_is_fitted

from regain.generalized_linear_model.ising import _fit
from regain.generalized_linear_model.ising import loss
from regain.covariance.time_graphical_lasso_ import init_precision
from regain.norm import l1_od_norm
from regain.utils import convergence
from regain.update_rules import update_rho
from regain.validation import check_norm_prox


def loss_ising(X, K, n_samples=None):
    """Loss function for time-varying ising model."""
    if n_samples is None:
        n_samples = np.ones(X.shape[0])
    return sum(-ni * loss(x, k) for x, k, ni in zip(X, K, n_samples))


def objective(X, K, Z_M, alpha, kernel, psi):
    """Objective function for time-varying ising model."""

    obj = loss_ising(X, K)
    obj += alpha * sum(map(l1_od_norm, K))

    for m in range(1, K.shape[0]):
        # all possible non markovians jumps
        Z_L, Z_R = Z_M[m]
        obj += np.sum(np.array(list(map(psi, Z_R - Z_L))) * np.diag(kernel, m))

    return obj


def _fit_time_ising_model(
    X,
    alpha=0.01,
    rho=1,
    kernel=None,
    max_iter=100,
    verbose=False,
    psi="laplacian",
    gamma=1e-3,
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
    n_cores=-1,
):
    """Time-varying graphical model solver.

    Solves the following problem via ADMM:
        min sum_{i=1}^T -n_i log_likelihood(K_i, X_i) + alpha ||K_i||_{od,1}
            + sum_{s>t}^T k(s,t) Psi(K_s - K_t)

    where X is a matrix n_i x D, the observations at time i and the
    log-likelihood changes according to the distribution.

    Parameters
    ----------
    X : ndarray, shape (n_times, n_samples, n_features)
        Data matrix. It has to contain two values: 0 or 1, -1 or 1.
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
    n_times, n_samples, n_features = X.shape
    n_samples = np.array([n_samples] * n_times)

    if kernel is None:
        kernel = np.eye(n_times)

    K = np.zeros((n_times, n_features, n_features))

    Z_M = {}
    U_M = {}
    Z_M_old = {}

    for m in range(1, n_times):
        # all possible non markovians jumps
        Z_L = K.copy()[:-m]
        Z_R = K.copy()[m:]
        Z_M[m] = (Z_L, Z_R)

        U_L = np.zeros_like(Z_L)
        U_R = np.zeros_like(Z_R)
        U_M[m] = (U_L, U_R)

        Z_L_old = np.zeros_like(Z_L)
        Z_R_old = np.zeros_like(Z_R)
        Z_M_old[m] = (Z_L_old, Z_R_old)

    checks = [convergence(obj=objective(X, K, Z_M, alpha, kernel, psi))]
    for iteration_ in range(max_iter):
        # update K

        A = np.zeros_like(K)
        for m in range(1, n_times):
            A[:-m] += Z_M[m][0] - U_M[m][0]
            A[m:] += Z_M[m][1] - U_M[m][1]

        A /= n_times
        A += A.transpose(0, 2, 1)
        A /= 2.0
        # K_new = np.zeros_like(K)

        print(K.shape)

        for t in range(n_times):
            K[t, :, :] = _fit(
                X=X[t, :, :],
                A=A[t, :, :],
                alpha=alpha,
                gamma=gamma,
                tol=tol,
                max_iter=max_iter,
                verbose=max(0, verbose - 1),
                compute_objective=True,
                warm_start=None,
                rho=rho,
                T=n_times,
                return_history=False,
                return_n_iter=False,
            )[0]

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
            sum(squared_norm(K[:-m] - Z_M[m][0]) + squared_norm(K[m:] - Z_M[m][1]) for m in range(1, n_times))
        )

        snorm = rho * np.sqrt(
            sum(
                squared_norm(Z_M[m][0] - Z_M_old[m][0]) + squared_norm(Z_M[m][1] - Z_M_old[m][1])
                for m in range(1, n_times)
            )
        )

        obj = objective(X, K, Z_M, alpha, kernel, psi) if compute_objective else np.nan

        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=n_features * n_times * tol
            + rtol
            * max(
                np.sqrt(sum(squared_norm(Z_M[m][0]) + squared_norm(Z_M[m][1]) for m in range(1, n_times))),
                np.sqrt(squared_norm(K) + sum(squared_norm(K[:-m]) + squared_norm(K[m:]) for m in range(1, n_times))),
            ),
            e_dual=n_features * n_times * tol
            + rtol * rho * np.sqrt(sum(squared_norm(U_M[m][0]) + squared_norm(U_M[m][1]) for m in range(1, n_times))),
        )
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
        # U_0 *= rho / rho_new
        for m in range(1, n_times):
            U_L, U_R = U_M[m]
            U_L *= rho / rho_new
            U_R *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [K]
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


class TemporalIsingModel(BaseEstimator):
    """Temporal Graphical model that follows an Ising model at each time point.

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

    compute_objective : boolean, default True
        Choose if compute the objective function during iterations
        (only useful if `verbose=True`).

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
        distribution="ising",
        alpha=0.01,
        kernel=None,
        rho=1.0,
        tol=1e-4,
        rtol=1e-4,
        psi="laplacian",
        max_iter=100,
        verbose=False,
        assume_centered=False,
        return_history=False,
        compute_objective=True,
        ker_param=1,
        max_iter_ext=100,
        n_cores=-1,
    ):
        self.alpha = alpha
        self.kernel = kernel
        self.rho = rho
        self.tol = tol
        self.rtol = rtol
        self.psi = psi
        self.max_iter = max_iter
        self.verbose = verbose
        self.assume_centered = assume_centered
        self.return_history = return_history
        self.compute_objective = compute_objective
        self.ker_param = ker_param
        self.max_iter_ext = max_iter_ext
        self.distribution = distribution
        self.n_cores = n_cores

    def get_precision(self):
        return self.precision_

    def fit(self, X, y):
        # Covariance does not make sense for a single feature
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order="C", ensure_min_features=2, estimator=self)

        self.classes_, n_samples = np.unique(y, return_counts=True)
        self.data = X.copy()
        if np.unique(self.data).size != 2:
            raise ValueError(
                "Using the ising distribution your data has " "to contain only two values, either 0 and 1 " "or -1, 1"
            )
        X = np.array([X[y == cl] for cl in self.classes_])

        if self.ker_param == "auto":
            from scipy.optimize import minimize_scalar

            if not callable(self.kernel):
                raise ValueError("kernel should be a function if ker_param=='auto'")
            # discover best kernel parameter via alternating minimization
            # initialise precision matrices, as warm start
            self.precision_ = init_precision(X, mode=self.init)
            theta_old = 0
            for i in range(self.max_iter_ext):
                # E step - discover best kernel parameter
                theta = minimize_scalar(
                    objective_kernel,
                    args=(self.precision_, self.psi, self.kernel, self.classes_[:, None]),
                    bounds=(0, X.shape[0]),
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

                out = _fit_time_ising_model(
                    X,
                    alpha=self.alpha,
                    rho=self.rho,
                    kernel=kernel,
                    tol=self.tol,
                    rtol=self.rtol,
                    psi=self.psi,
                    max_iter=self.max_iter,
                    verbose=self.verbose,
                    return_n_iter=True,
                    return_history=self.return_history,
                    compute_objective=self.compute_objective,
                    n_cores=self.n_cores,
                )
                if self.return_history:
                    self.precision_, self.history_, self.n_iter_ = out
                else:
                    self.precision_, self.n_iter_ = out
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
            out = _fit_time_ising_model(
                X,
                alpha=self.alpha,
                rho=self.rho,
                kernel=kernel,
                tol=self.tol,
                rtol=self.rtol,
                psi=self.psi,
                max_iter=self.max_iter,
                verbose=self.verbose,
                return_n_iter=True,
                return_history=self.return_history,
                compute_objective=self.compute_objective,
            )
            if self.return_history:
                self.precision_, self.history_, self.n_iter_ = out
            else:
                self.precision_, self.n_iter_ = out

        return self

    def score(self, X, y):
        """Computes the log-likelihood of a Gaussian data set with
        `self.covariance_` as an estimator of its covariance matrix.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X is assumed to be drawn from the same distribution than
            the data used in fit (including centering).
        y :  array-like, shape = (n_samples,)
            Class of samples.
        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.
        """
        # Covariance does not make sense for a single feature
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order="C", ensure_min_features=2, estimator=self)

        # TO THINK
        return -99999999


def precision_similarity(precision, psi=None):
    from regain.norm import l1_od_norm
    from scipy.spatial.distance import squareform
    from itertools import combinations

    distances = squareform([l1_od_norm(t1 - t2) for t1, t2 in combinations(precision, 2)])
    print(distances)
    distances /= np.max(distances)
    return 1 - distances


class SimilarityTemporalIsingModel(TemporalIsingModel):
    """Learn how to relate different adjacencies matrices across times.

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

    compute_objective : boolean, default True
        Choose if compute the objective function during iterations
        (only useful if `verbose=True`).

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
        compute_objective=True,
        ker_param=1,
        max_iter_ext=100,
        eps=1e-6,
        n_clusters=None,
    ):
        super(SimilarityTemporalIsingModel, self).__init__(
            alpha=alpha,
            rho=rho,
            tol=tol,
            rtol=rtol,
            max_iter=max_iter,
            verbose=verbose,
            assume_centered=assume_centered,
            compute_objective=compute_objective,
            return_history=return_history,
            psi=psi,
        )
        # in this class, `kernel` is either a matrix TxT or None
        # if None, automatically learn all the weights
        self.beta = beta
        self.kernel = kernel
        self.max_iter_ext = max_iter_ext
        self.eps = eps
        self.n_clusters = n_clusters

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order="C", ensure_min_features=2, estimator=self)

        self.classes_, n_samples = np.unique(y, return_counts=True)
        self.data = X.copy()
        if np.unique(self.data).size != 2:
            raise ValueError(
                "Using the ising distribution your data has " "to contain only two values, either 0 and 1 " "or -1, 1"
            )
        X = np.array([X[y == cl] for cl in self.classes_])
        if self.kernel is None:
            # from scipy.optimize import minimize
            # discover best kernel parameter via EM
            # initialise precision matrices, as warm start
            self.precision_ = np.random.rand(self.classes_.shape[0], X.shape[1], X.shape[1])
            n_times = self.precision_.shape[0]
            kernel = np.eye(n_times)

            psi, _, _ = check_norm_prox(self.psi)
            if self.n_clusters is None:
                self.n_clusters = n_times
            labels_pred_old = 0
            for i in range(self.max_iter_ext):
                out = _fit_time_ising_model(
                    X,
                    alpha=self.alpha,
                    rho=self.rho,
                    kernel=kernel,
                    tol=self.tol,
                    rtol=self.rtol,
                    psi=self.psi,
                    max_iter=self.max_iter,
                    verbose=self.verbose,
                    return_n_iter=True,
                    return_history=self.return_history,
                    compute_objective=self.compute_objective,
                )
                if self.return_history:
                    (self.precision_, self.history_, self.n_iter_) = out
                else:
                    self.precision_, self.n_iter_ = out
                theta = precision_similarity(self.precision_, psi)
                kernel = theta
                labels_pred = AgglomerativeClustering(
                    n_clusters=self.n_clusters, affinity="precomputed", linkage="complete"
                ).fit_predict(kernel)

                if i > 0 and np.linalg.norm(labels_pred - labels_pred_old) / labels_pred.size < self.eps:
                    break
                kernel = kernels.RBF(0.0001)(labels_pred[:, None]) + kernels.RBF(self.beta)(
                    np.arange(n_times)[:, None]
                )
                print(kernel)
                labels_pred_old = labels_pred

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

            out = _fit_time_ising_model(
                X,
                alpha=self.alpha,
                rho=self.rho,
                kernel=kernel,
                tol=self.tol,
                rtol=self.rtol,
                psi=self.psi,
                max_iter=self.max_iter,
                verbose=self.verbose,
                return_n_iter=True,
                return_history=self.return_history,
                compute_objective=self.compute_objective,
            )
            if self.return_history:
                (self.precision_, self.history_, self.n_iter_) = out
            else:
                self.precision_, self.n_iter_ = out

        return self

    def transform(self, X, y=None):
        """Possibility to add in a Pipeline."""
        check_is_fitted(self, ["similarity_matrix_"])

        return self.similarity_matrix_
