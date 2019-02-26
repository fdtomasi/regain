"""Sparse inverse covariance selection over time via ADMM.

This adds the possibility to specify a temporal constraint with a kernel
function.
"""
from __future__ import division

import warnings

import numpy as np
from scipy import linalg
from six.moves import map, range, zip
from sklearn.covariance import empirical_covariance, log_likelihood
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_X_y

from regain.covariance.time_graph_lasso_ import TimeGraphLasso, loss
from regain.norm import l1_od_norm
from regain.prox import prox_logdet, soft_thresholding
from regain.update_rules import update_rho
from regain.utils import convergence
from regainpr.validation import check_norm_prox


def objective(n_samples, S, K, Z_0, Z_M, alpha, kernel, psi):
    """Objective function for time-varying graphical lasso."""
    obj = loss(S, K, n_samples=n_samples)
    obj += alpha * sum(map(l1_od_norm, Z_0))

    for m in range(1, Z_0.shape[0]):
        # all possible markovians jumps
        Z_L, Z_R = Z_M[m]
        obj += np.sum(np.array(list(map(psi, Z_R - Z_L))) * np.diag(kernel, m))

    return obj


def kernel_time_graphical_lasso(
        emp_cov, alpha=0.01, rho=1, kernel=None, max_iter=100, n_samples=None,
        verbose=False, psi='laplacian', tol=1e-4, rtol=1e-4,
        return_history=False, return_n_iter=True, mode='admm',
        update_rho_options=None, compute_objective=True, stop_at=None,
        stop_when=1e-4):
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

    covariance_ = emp_cov.copy()
    covariance_ *= 0.95
    K = np.empty_like(emp_cov)
    for i, (c, e) in enumerate(zip(covariance_, emp_cov)):
        c.flat[::n_features + 1] = e.flat[::n_features + 1]
        K[i] = linalg.pinvh(c)

    # K = np.zeros_like(emp_cov)
    Z_0 = K.copy()  # np.zeros_like(emp_cov)
    U_0 = np.zeros_like(Z_0)
    Z_0_old = np.zeros_like(Z_0)

    Z_M = {}
    U_M = {}
    Z_M_old = {}
    for m in range(1, n_times):
        # all possible markovians jumps
        Z_L = K.copy()[:-m]
        Z_R = K.copy()[m:]
        Z_M[m] = (Z_L, Z_R)

        U_L = np.zeros_like(Z_L)
        U_R = np.zeros_like(Z_R)
        U_M[m] = (U_L, U_R)

        Z_L_old = np.zeros_like(Z_L)
        Z_R_old = np.zeros_like(Z_R)
        Z_M_old[m] = (Z_L_old, Z_R_old)

    if n_samples is None:
        n_samples = np.ones(n_times)

    checks = [
        convergence(
            obj=objective(n_samples, emp_cov, Z_0, K, Z_M, alpha, kernel, psi))
    ]
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
        A /= 2.

        A *= -rho * n_times / n_samples[:, None, None]
        A += emp_cov

        K = np.array(
            [
                prox_logdet(a, lamda=ni / (rho * n_times))
                for a, ni in zip(A, n_samples)
            ])

        # update Z_0
        A = K + U_0
        A += A.transpose(0, 2, 1)
        A /= 2.
        Z_0 = soft_thresholding(A, lamda=alpha / rho)

        # update residuals
        U_0 += K - Z_0

        # other Zs
        for m in range(1, n_times):
            U_L, U_R = U_M[m]
            A_L = K[:-m] + U_L
            A_R = K[m:] + U_R
            if not psi_node_penalty:
                prox_e = prox_psi(
                    A_R - A_L,
                    lamda=2. * np.diag(kernel, m)[:, None, None] / rho)
                Z_L = .5 * (A_L + A_R - prox_e)
                Z_R = .5 * (A_L + A_R + prox_e)
            else:
                Z_L, Z_R = prox_psi(
                    np.concatenate((A_L, A_R), axis=1),
                    lamda=.5 * np.diag(kernel, m)[:, None, None] / rho,
                    rho=rho, tol=tol, rtol=rtol, max_iter=max_iter)
            Z_M[m] = (Z_L, Z_R)

            # update other residuals
            U_L += K[:-m] - Z_L
            U_R += K[m:] - Z_R

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(K - Z_0) + sum(
                squared_norm(K[:-m] - Z_M[m][0]) +
                squared_norm(K[m:] - Z_M[m][1]) for m in range(1, n_times)))

        snorm = rho * np.sqrt(
            squared_norm(Z_0 - Z_0_old) + sum(
                squared_norm(Z_M[m][0] - Z_M_old[m][0]) +
                squared_norm(Z_M[m][1] - Z_M_old[m][1])
                for m in range(1, n_times)))

        obj = objective(
            n_samples, emp_cov, Z_0, K, Z_M, alpha, kernel, psi) \
            if compute_objective else np.nan

        check = convergence(
            obj=obj, rnorm=rnorm, snorm=snorm,
            e_pri=n_features * n_times * tol + rtol * max(
                np.sqrt(
                    squared_norm(Z_0) + sum(
                        squared_norm(Z_M[m][0]) + squared_norm(Z_M[m][1])
                        for m in range(1, n_times))),
                np.sqrt(
                    squared_norm(K) + sum(
                        squared_norm(K[:-m]) + squared_norm(K[m:])
                        for m in range(1, n_times)))),
            e_dual=n_features * n_times * tol + rtol * rho * np.sqrt(
                squared_norm(U_0) + sum(
                    squared_norm(U_M[m][0]) + squared_norm(U_M[m][1])
                    for m in range(1, n_times))))
        Z_0_old = Z_0.copy()
        for m in range(1, n_times):
            Z_M_old[m] = (Z_M[m][0].copy(), Z_M[m][1].copy())

        if verbose:
            print(
                "obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        checks.append(check)
        if stop_at is not None:
            if abs(check.obj - stop_at) / abs(stop_at) < stop_when:
                break

        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        rho_new = update_rho(
            rho, rnorm, snorm, iteration=iteration_,
            **(update_rho_options or {}))
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


class KernelTimeGraphLasso_(TimeGraphLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

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
    covariance_ : array-like, shape (n_times, n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_times, n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(
            self, alpha=0.01, kernel=None, mode='admm', rho=1.,
            time_on_axis='first', tol=1e-4, rtol=1e-4, psi='laplacian',
            max_iter=100, verbose=False, assume_centered=False,
            return_history=False, update_rho_options=None,
            compute_objective=True, stop_at=None, stop_when=1e-4,
            suppress_warn_list=False):
        super(KernelTimeGraphLasso_, self).__init__(
            alpha=alpha, rho=rho, tol=tol, rtol=rtol, max_iter=max_iter,
            verbose=verbose, assume_centered=assume_centered, mode=mode,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective,
            suppress_warn_list=suppress_warn_list, stop_at=stop_at,
            stop_when=stop_when, return_history=return_history,
            time_on_axis=time_on_axis, psi=psi)
        self.kernel = kernel

    def _fit(self, emp_cov, n_samples):
        """Fit the TimeGraphLasso model to X.

        Parameters
        ----------
        emp_cov : ndarray, shape (n_time, n_features, n_features)
            Empirical covariance of data.

        """
        out = kernel_time_graphical_lasso(
            emp_cov, alpha=self.alpha, rho=self.rho, kernel=self.kernel,
            mode=self.mode, n_samples=n_samples, tol=self.tol, rtol=self.rtol,
            psi=self.psi, max_iter=self.max_iter, verbose=self.verbose,
            return_n_iter=True, return_history=self.return_history,
            update_rho_options=self.update_rho_options,
            compute_objective=self.compute_objective, stop_at=self.stop_at,
            stop_when=self.stop_when)
        if self.return_history:
            self.precision_, self.covariance_, self.history_, self.n_iter_ = out
        else:
            self.precision_, self.covariance_, self.n_iter_ = out
        return self


class KernelTimeGraphicalLasso(TimeGraphLasso):
    """As KernelTimeGraphLasso, but X is 2d and y specifies time.

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
    covariance_ : array-like, shape (n_times, n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_times, n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(
            self, alpha=0.01, kernel=None, rho=1., tol=1e-4, rtol=1e-4,
            psi='laplacian', max_iter=100, verbose=False,
            assume_centered=False, return_history=False,
            update_rho_options=None, compute_objective=True, ker_param=1):
        super(KernelTimeGraphicalLasso, self).__init__(
            alpha=alpha, rho=rho, tol=tol, rtol=rtol, max_iter=max_iter,
            verbose=verbose, assume_centered=assume_centered,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective, return_history=return_history,
            psi=psi)
        self.kernel = kernel
        self.ker_param = ker_param

    def _fit(self, emp_cov, n_samples):
        if callable(self.kernel):
            try:
                # this works if it is a ExpSineSquared or RBF kernel
                kernel = self.kernel(length_scale=self.ker_param)(
                    self.classes_[:, None])
            except TypeError:
                # maybe it's a ConstantKernel
                kernel = self.kernel(constant_value=self.ker_param)(
                    self.classes_[:, None])
        else:
            kernel = self.kernel
            if kernel.shape[0] != self.classes_.size:
                raise ValueError(
                    "Kernel size does not match classes of samples, "
                    "got {} classes and kernel has shape {}".format(
                        self.classes_.size, kernel.shape[0]))

        out = kernel_time_graphical_lasso(
            emp_cov, alpha=self.alpha, rho=self.rho, kernel=kernel,
            n_samples=n_samples, tol=self.tol, rtol=self.rtol, psi=self.psi,
            max_iter=self.max_iter, verbose=self.verbose, return_n_iter=True,
            return_history=self.return_history,
            update_rho_options=self.update_rho_options,
            compute_objective=self.compute_objective)
        if self.return_history:
            self.precision_, self.covariance_, self.history_, self.n_iter_ = out
        else:
            self.precision_, self.covariance_, self.n_iter_ = out

        return self

    def fit(self, X, y):
        """Fit the KernelTimeGraphLasso model to X.

        Parameters
        ----------
        X : ndarray, shape = (n_samples * n_times, n_dimensions)
            Data matrix.
        y : ndarray, shape = (n_times,)
            Indicate the temporal belonging of each sample.
        """
        # Covariance does not make sense for a single feature
        X, y = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, order="C",
            ensure_min_features=2, estimator=self)

        n_dimensions = X.shape[1]
        self.classes_, n_samples = np.unique(y, return_counts=True)
        n_times = self.classes_.size

        # n_samples = np.array([x.shape[0] for x in X])
        if self.assume_centered:
            self.location_ = np.zeros((n_times, n_dimensions))
        else:
            self.location_ = np.array(
                [X[y == cl].mean(0) for cl in self.classes_])

        emp_cov = np.array(
            [
                empirical_covariance(
                    X[y == cl], assume_centered=self.assume_centered)
                for cl in self.classes_
            ])

        return self._fit(emp_cov, n_samples)

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
        X, y = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, order="C",
            ensure_min_features=2, estimator=self)

        # compute empirical covariance of the test set
        test_cov = np.array(
            [
                empirical_covariance(
                    X[y == cl] - self.location_[i], assume_centered=True)
                for i, cl in enumerate(self.classes_)
            ])

        res = sum(
            X[y == cl].shape[0] * log_likelihood(S, K) for S, K, n in zip(
                test_cov, self.get_observed_precision(), self.classes_))

        return -99999999 if res == -np.inf else res
