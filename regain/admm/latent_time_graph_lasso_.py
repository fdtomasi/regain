"""Graphical latent variable models selection over time via ADMM."""
from __future__ import division

import numpy as np
import warnings

from functools import partial
from six.moves import range
from sklearn.covariance import empirical_covariance, log_likelihood
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.extmath import squared_norm, fast_logdet
from sklearn.utils.validation import check_array

from regain.admm.time_graph_lasso_ import log_likelihood as logl
from regain.admm.time_graph_lasso_ import log_likelihood_trace
from regain.norm import l1_od_norm
from regain.prox import soft_thresholding_sign
from regain.prox import prox_logdet
from regain.prox import prox_trace_indicator
from regain.utils import convergence, error_norm_time
from regain.validation import check_norm_prox


def objective(S, R, Z_0, Z_1, Z_2, W_0, W_1, W_2,
              alpha, tau, beta, eta, psi, phi):
    """Objective function for time-varying graphical lasso."""
    obj = np.sum(- logl(s, r) for s, r in zip(S, R))
    obj += alpha * np.sum(map(l1_od_norm, Z_0))
    obj += tau * np.sum(map(partial(np.linalg.norm, ord='nuc'), W_0))
    obj += beta * np.sum(map(psi, Z_2 - Z_1))
    obj += eta * np.sum(map(phi, W_2 - W_1))
    return obj


def latent_time_graph_lasso(
        emp_cov, alpha=1, tau=1, rho=1, beta=1., eta=1., max_iter=1000,
        verbose=False, psi='laplacian', phi='laplacian', mode=None,
        tol=1e-4, rtol=1e-2, assume_centered=False,
        return_history=False, return_n_iter=True):
    r"""Time-varying latent variable graphical lasso solver.

    Solves the following problem via ADMM:
        min sum_{i=1}^T -n_i log_likelihood(K_i-L_i) + alpha ||K_i||_{od,1}
            + tau ||L_i||_*
            + beta sum_{i=2}^T Psi(K_i - K_{i-1})
            + eta sum_{i=2}^T Phi(L_i - L_{i-1})

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    data_list : list of 2-dimensional matrices.
        Input matrices.
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

    K = np.zeros_like(emp_cov)
    Z_0 = np.zeros_like(K)
    Z_1 = np.zeros_like(K)[:-1]
    Z_2 = np.zeros_like(K)[1:]
    W_0 = np.zeros_like(K)
    W_1 = np.zeros_like(K)[:-1]
    W_2 = np.zeros_like(K)[1:]
    X_0 = np.zeros_like(K)
    X_1 = np.zeros_like(K)[:-1]
    X_2 = np.zeros_like(K)[1:]
    U_1 = np.zeros_like(K)[:-1]
    U_2 = np.zeros_like(K)[1:]

    R_old = np.zeros_like(K)
    Z_1_old = np.zeros_like(Z_1)
    Z_2_old = np.zeros_like(Z_2)
    W_1_old = np.zeros_like(W_1)
    W_2_old = np.zeros_like(W_2)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.full(K.shape[0], 3, dtype=float)
    divisor[0] -= 1
    divisor[-1] -= 1

    checks = []
    for iteration_ in range(max_iter):
        # update R
        A = Z_0 - W_0 - X_0
        A *= - rho
        A += emp_cov
        R = np.array(map(partial(prox_logdet, lamda=1. / rho), A))

        # update Z_0
        A = R + W_0 + X_0
        A[:-1] += Z_1 - X_1
        A[1:] += Z_2 - X_2
        A /= divisor[:, None, None]
        soft_thresholding = partial(soft_thresholding_sign, lamda=alpha / rho)
        Z_0 = np.array(map(soft_thresholding, A))

        # update Z_1, Z_2
        A_1 = Z_0[:-1] + X_1
        A_2 = Z_0[1:] + X_2
        if not psi_node_penalty:
            prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)
            Z_1 = .5 * (A_1 + A_2 - prox_e)
            Z_2 = .5 * (A_1 + A_2 + prox_e)
        else:
            Z_1, Z_2 = prox_psi(np.concatenate((A_1, A_2), axis=1),
                                lamda=.5 * beta / rho,
                                rho=rho, tol=tol, rtol=rtol, max_iter=max_iter)

        # update W_0
        A = Z_0 - R - X_0
        A[:-1] += W_1 - U_1
        A[1:] += W_2 - U_2
        A /= divisor[:, None, None]
        W_0 = np.array(map(partial(prox_trace_indicator, lamda=tau / rho), A))

        # update W_1, W_2
        A_1 = W_0[:-1] + U_1
        A_2 = W_0[1:] + U_2
        if not phi_node_penalty:
            prox_e = prox_phi(A_2 - A_1, lamda=2. * eta / rho)
            W_1 = .5 * (A_1 + A_2 - prox_e)
            W_2 = .5 * (A_1 + A_2 + prox_e)
        else:
            W_1, W_2 = prox_phi(np.concatenate((A_1, A_2), axis=1),
                                lamda=.5 * eta / rho,
                                rho=rho, tol=tol, rtol=rtol, max_iter=max_iter)

        # update residuals
        X_0 += R - Z_0 + W_0
        X_1 += Z_0[:-1] - Z_1
        X_2 += Z_0[1:] - Z_2
        U_1 += W_0[:-1] - W_1
        U_2 += W_0[1:] - W_2

        # diagnostics, reporting, termination checks
        # X_consensus = np.zeros_like(X_0)
        # X_consensus[:-1] += X_1
        # X_consensus[1:] += X_2
        # X_consensus /= divisor[:, None, None] - 1
        #
        # U_consensus = np.zeros_like(X_0)
        # U_consensus[:-1] += U_1
        # U_consensus[1:] += U_2
        # U_consensus /= divisor[:, None, None] - 1
        #
        # Z_consensus = np.zeros_like(Z_0)
        # Z_consensus[:-1] += Z_1
        # Z_consensus[1:] += Z_2
        # Z_consensus /= divisor[:, None, None] - 1
        #
        # W_consensus = np.zeros_like(W_0)
        # W_consensus[:-1] += W_1
        # W_consensus[1:] += W_2
        # W_consensus /= divisor[:, None, None] - 1

        check = convergence(
            obj=objective(emp_cov, R, Z_0, Z_1, Z_2, W_0, W_1, W_2,
                          alpha, tau, beta, eta, psi, phi),
            rnorm=np.sqrt(squared_norm(R - Z_0 + W_0) +
                          squared_norm(Z_0[:-1] - Z_1) +
                          squared_norm(Z_0[1:] - Z_2) +
                          squared_norm(W_0[:-1] - W_1) +
                          squared_norm(W_0[1:] - W_2)),
            snorm=np.sqrt(squared_norm(rho * (R - R_old)) +
                          squared_norm(rho * (Z_1 - Z_1_old)) +
                          squared_norm(rho * (Z_2 - Z_2_old)) +
                          squared_norm(rho * (W_1 - W_1_old)) +
                          squared_norm(rho * (W_2 - W_2_old))),
            e_pri=np.sqrt(np.prod(K.shape[1:]) * (5 * K.shape[0] - 4)) * tol +
                  rtol * max(
                np.sqrt(squared_norm(R) +
                        squared_norm(Z_1) + squared_norm(Z_2) +
                        squared_norm(W_1) + squared_norm(W_2)),
                np.sqrt(squared_norm(Z_0) - squared_norm(W_0) +
                        squared_norm(Z_0[:-1]) + squared_norm(Z_0[1:]) +
                        squared_norm(W_0[:-1]) + squared_norm(W_0[1:]))),
            e_dual=np.sqrt(np.prod(K.shape[1:]) * (5 * K.shape[0] - 4)) * tol +
                   rtol * np.sqrt(
                squared_norm(rho * X_0) +
                squared_norm(rho * X_1) + squared_norm(rho * X_2) +
                squared_norm(rho * U_1) + squared_norm(rho * U_2)))

        R_old = R.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()
        W_1_old = W_1.copy()
        W_2_old = W_2.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        # if iteration_ % 10 == 0 and rho > 1e-6:
        #     rho /= 2.
    else:
        warnings.warn("Objective did not converge.")

    return_list = [Z_0, W_0, emp_cov]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class LatentTimeGraphLasso(EmpiricalCovariance):
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

    def __init__(self, alpha=1., tau=1., beta=1., eta=1., mode='cd', rho=1.,
                 bypass_transpose=True, tol=1e-4, rtol=1e-4,
                 psi='laplacian', phi='laplacian', max_iter=100,
                 verbose=False, assume_centered=False):
        super(LatentTimeGraphLasso, self).__init__(assume_centered=assume_centered)
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.eta = eta
        self.rho = rho
        self.mode = mode
        self.tol = tol
        self.rtol = rtol
        self.psi = psi
        self.phi = phi
        self.max_iter = max_iter
        self.verbose = verbose
        # for splitting purposes, data may come transposed, with time in the
        # last index. Set bypass_transpose=True if X comes with time in the
        # first dimension already
        self.bypass_transpose = bypass_transpose

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
        self.precision_, self.latent_, self.covariance_, self.n_iter_ = \
            latent_time_graph_lasso(
                emp_cov, alpha=self.alpha, tau=self.tau, rho=self.rho,
                beta=self.beta, eta=self.eta, mode=self.mode,
                tol=self.tol, rtol=self.rtol, psi=self.psi, phi=self.phi,
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

        res = np.sum([log_likelihood(S, R) for S, R in zip(
            test_cov, self.precision_ - self.latent_)])

        # ALLA  MATLAB1
        # ranks = [np.linalg.matrix_rank(L) for L in self.latent_]
        # scores_ranks = np.square(ranks-np.sqrt(L.shape[1]))

        return res # - np.sum(scores_ranks)

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
        precision = self.get_precision()
        # compute mahalanobis distances
        sum_ = 0.
        for obs, loc in zip(observations, self.location_):
            centered_obs = observations - self.location_
            sum_ += np.sum(
                np.dot(centered_obs, precision) * centered_obs, 1)

        mahalanobis_dist = sum_ / len(observations)
        return mahalanobis_dist
