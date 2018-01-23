"""Sparse inverse covariance selection over time via ADMM.

More information can be found in the paper linked at:
[cite Hallac]
"""
from __future__ import division

import numpy as np
import warnings

from six.moves import range, map, zip
from sklearn.covariance import empirical_covariance
from sklearn.covariance import log_likelihood
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_array

from regain.admm.graph_lasso_ import GraphLasso, logl
from regain.norm import l1_od_norm
from regain.prox import prox_logdet
from regain.prox import soft_thresholding_sign
from regain.update_rules import update_rho
from regain.utils import convergence, error_norm_time
from regain.validation import check_norm_prox


def objective(S, K, Z_0, Z_1, Z_2, alpha, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = sum(- logl(emp_cov, precision) for emp_cov, precision in zip(S, K))
    obj += alpha * sum(map(l1_od_norm, Z_0))
    obj += beta * sum(map(psi, Z_2 - Z_1))
    return obj


def time_graph_lasso(
        emp_cov, alpha=1, rho=1, beta=1, max_iter=100,
        verbose=False, psi='laplacian', tol=1e-4, rtol=1e-2,
        return_history=False, return_n_iter=True, mode=None):
    """Time-varying graphical lasso solver.

    Solves the following problem via ADMM:
        minimize  trace(S*X) - log det X + lambda*||X||_1

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    data_list : list of 2-dimensional matrices.
        Input matrices.
    lamda : float, optional
        Regularisation parameter.
    rho : float, optional
        Augmented Lagrangian parameter.
    alpha : float, optional
        Over-relaxation parameter (typically between 1.0 and 1.8).
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

    K = np.zeros_like(emp_cov)
    Z_0 = np.zeros_like(emp_cov)
    Z_1 = np.zeros_like(emp_cov)[:-1]
    Z_2 = np.zeros_like(emp_cov)[1:]

    U_0 = np.zeros_like(Z_0)
    U_1 = np.zeros_like(Z_1)
    U_2 = np.zeros_like(Z_2)

    Z_0_old = np.zeros_like(Z_0)
    Z_1_old = np.zeros_like(Z_1)
    Z_2_old = np.zeros_like(Z_2)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.full(emp_cov.shape[0], 3, dtype=float)
    divisor[0] -= 1
    divisor[-1] -= 1

    checks = []
    for iteration_ in range(max_iter):
        # x-update
        A = K + U_0
        A *= - rho
        A += emp_cov
        Z_0 = np.array([prox_logdet(a_, lamda=1. / rho) for a_ in A])

        # z-update with relaxation
        A = Z_0 - U_0
        A[:-1] += Z_1 - U_1
        A[1:] += Z_2 - U_2
        A /= divisor[:, None, None]
        # soft_thresholding_ = partial(soft_thresholding, lamda=alpha / rho)
        # K = np.array(map(soft_thresholding_, A))
        K = soft_thresholding_sign(A, lamda=alpha / rho)

        # other Zs
        A_1 = K[:-1] + U_1
        A_2 = K[1:] + U_2
        if not psi_node_penalty:
            prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)
            Z_1 = .5 * (A_1 + A_2 - prox_e)
            Z_2 = .5 * (A_1 + A_2 + prox_e)
        else:
            Z_1, Z_2 = prox_psi(np.concatenate((A_1, A_2), axis=1),
                                lamda=.5 * beta / rho,
                                rho=rho, tol=tol, rtol=rtol, max_iter=max_iter)

        # update residuals
        U_0 += K - Z_0
        U_1 += K[:-1] - Z_1
        U_2 += K[1:] - Z_2

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(squared_norm(K - Z_0) +
                        squared_norm(K[:-1] - Z_1) +
                        squared_norm(K[1:] - Z_2))

        snorm = rho * np.sqrt(squared_norm(Z_0 - Z_0_old) +
                              squared_norm(Z_1 - Z_1_old) +
                              squared_norm(Z_2 - Z_2_old))

        check = convergence(
            obj=objective(emp_cov, Z_0, K, Z_1, Z_2, alpha, beta, psi),
            rnorm=rnorm, snorm=snorm,
            e_pri=np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * max(
                np.sqrt(squared_norm(Z_0) + squared_norm(Z_1) + squared_norm(Z_2)),
                np.sqrt(squared_norm(K) + squared_norm(K[:-1]) + squared_norm(K[1:]))),
            e_dual=np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * rho * np.sqrt(
                squared_norm(U_0) + squared_norm(U_1) + squared_norm(U_2))
        )
        Z_0_old = Z_0.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_)
        # scaled dual variables should be also rescaled
        U_0 *= rho / rho_new
        U_1 *= rho / rho_new
        U_2 *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [K, emp_cov]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class TimeGraphLasso(GraphLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    Parameters
    ----------
    alpha : positive float, default 0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

    mode : {'cd', 'lars'}, default 'cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : positive float, default 1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped.

    enet_tol : positive float, optional
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'.

    max_iter : integer, default 100
        The maximum number of iterations.

    verbose : boolean, default False
        If verbose is True, the objective function and dual gap are
        plotted at each iteration.

    assume_centered : boolean, default False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    graph_lasso, GraphLassoCV

    """

    def __init__(self, alpha=.01, beta=1., mode='cd', rho=1.,
                 bypass_transpose=True, tol=1e-4, rtol=1e-4,
                 psi='laplacian', max_iter=100,
                 verbose=False, assume_centered=False):
        super(TimeGraphLasso, self).__init__(
            alpha=alpha, rho=rho, tol=tol, rtol=rtol, max_iter=max_iter,
            verbose=verbose, assume_centered=assume_centered)
        self.beta = beta
        self.psi = psi
        self.mode = mode
        # for splitting purposes, data may come transposed, with time in the
        # last index. Set bypass_transpose=True if X comes with time in the
        # first dimension already
        self.bypass_transpose = bypass_transpose

    def get_observed_precision(self):
        """Getter for the observed precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.

        """
        return self.get_precision()

    def fit(self, X, y=None):
        """Fit the GraphLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_time, n_samples, n_features), or
                (n_samples, n_features, n_time)
            Data from which to compute the covariance estimate.
            If shape is (n_samples, n_features, n_time), then set
            `bypass_transpose = False`.
        y : (ignored)
        """
        if not self.bypass_transpose:
            X = X.transpose(2, 0, 1)  # put time as first dimension
        # Covariance does not make sense for a single feature
        # X = check_array(X, allow_nd=True, estimator=self)
        # if X.ndim != 3:
        #     raise ValueError("Found array with dim %d. %s expected <= 2."
        #                      % (X.ndim, self.__class__.__name__))
        X = np.array([check_array(x, ensure_min_features=2,
                      ensure_min_samples=2, estimator=self) for x in X])

        if self.assume_centered:
            self.location_ = np.zeros((X.shape[0], 1, X.shape[2]))
        else:
            self.location_ = X.mean(1).reshape(X.shape[0], 1, X.shape[2])
        emp_cov = np.array([empirical_covariance(
            x, assume_centered=self.assume_centered) for x in X])
        self.precision_, self.covariance_, self.n_iter_ = \
            time_graph_lasso(
                emp_cov, alpha=self.alpha, rho=self.rho,
                beta=self.beta, mode=self.mode,
                tol=self.tol, rtol=self.rtol, psi=self.psi,
                max_iter=self.max_iter, verbose=self.verbose,
                return_n_iter=True, return_history=False)
        return self

    def score(self, X_test, y=None):
        """Computes the log-likelihood of a Gaussian data set with
        `self.covariance_` as an estimator of its covariance matrix.

        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y : not used, present for API consistence purpose.

        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.

        """
        if not self.bypass_transpose:
            X_test = X_test.transpose(2, 0, 1)  # put time as first dimension
        # compute empirical covariance of the test set
        test_cov = np.array([empirical_covariance(
            x, assume_centered=True) for x in X_test - self.location_])

        res = sum(log_likelihood(S, K) for S, K in zip(
            test_cov, self.get_observed_precision()))

        # ALLA  MATLAB1
        # ranks = [np.linalg.matrix_rank(L) for L in self.latent_]
        # scores_ranks = np.square(ranks-np.sqrt(L.shape[1]))

        return res  # - np.sum(scores_ranks)

    def error_norm(self, comp_cov, norm='frobenius', scaling=True,
                   squared=True):
        """Compute the Mean Squared Error between two covariance estimators.
        (In the sense of the Frobenius norm).

        Parameters
        ----------
        comp_cov : array-like, shape = [n_features, n_features]
            The covariance to compare with.

        norm : str
            The type of norm used to compute the error. Available error types:
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            where A is the error ``(comp_cov - self.covariance_)``.

        scaling : bool
            If True (default), the squared error norm is divided by n_features.
            If False, the squared error norm is not rescaled.

        squared : bool
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.

        Returns
        -------
        The Mean Squared Error (in the sense of the Frobenius norm) between
        `self` and `comp_cov` covariance estimators.

        """
        return error_norm_time(self.covariance_, comp_cov, norm=norm,
                               scaling=scaling, squared=squared)

    def mahalanobis(self, observations):
        """Computes the squared Mahalanobis distances of given observations.

        Parameters
        ----------
        observations : array-like, shape = [n_observations, n_features]
            The observations, the Mahalanobis distances of the which we
            compute. Observations are assumed to be drawn from the same
            distribution than the data used in fit.

        Returns
        -------
        mahalanobis_distance : array, shape = [n_observations,]
            Squared Mahalanobis distances of the observations.

        """
        if not self.bypass_transpose:
            # put time as first dimension
            observations = observations.transpose(2, 0, 1)
        precision = self.get_observed_precision()
        # compute mahalanobis distances
        sum_ = 0.
        for obs, loc in zip(observations, self.location_):
            centered_obs = observations - self.location_
            sum_ += np.sum(
                np.dot(centered_obs, precision) * centered_obs, 1)

        mahalanobis_dist = sum_ / len(observations)
        return mahalanobis_dist
