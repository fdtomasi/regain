"""Sparse inverse covariance selection over time via ADMM.

More information can be found in the paper linked at:
https://arxiv.org/abs/1703.01958
"""
from __future__ import division

import warnings

import numpy as np
from scipy import linalg
from six.moves import map, range, zip
from sklearn.covariance import empirical_covariance, log_likelihood
from sklearn.utils.extmath import squared_norm

from regain.covariance.graphical_lasso_ import GraphicalLasso, logl
from regain.norm import l1_od_norm
from regain.prox import prox_logdet, soft_thresholding
from regain.update_rules import update_rho
from regain.utils import convergence, error_norm_time
from regain.validation import check_input, check_norm_prox


def loss(S, K, n_samples=None):
    """Loss function for time-varying graphical lasso."""
    if n_samples is None:
        n_samples = np.ones(S.shape[0])
    return sum(
        -ni * logl(emp_cov, precision)
        for emp_cov, precision, ni in zip(S, K, n_samples))


def objective(n_samples, S, K, Z_0, Z_1, Z_2, alpha, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = loss(S, K, n_samples=n_samples)

    if isinstance(alpha, np.ndarray):
        obj += sum(l1_od_norm(a * z) for a, z in zip(alpha, Z_0))
    else:
        obj += alpha * sum(map(l1_od_norm, Z_0))

    if isinstance(beta, np.ndarray):
        obj += sum(b[0][0] * m for b, m in zip(beta, map(psi, Z_2 - Z_1)))
    else:
        obj += beta * sum(map(psi, Z_2 - Z_1))
    return obj


def time_graphical_lasso(
        emp_cov, alpha=0.01, rho=1, beta=1, max_iter=100, n_samples=None,
        verbose=False, psi='laplacian', tol=1e-4, rtol=1e-4,
        return_history=False, return_n_iter=True, mode='admm',
        update_rho_options=None, compute_objective=True, stop_at=None,
        stop_when=1e-4, init='empirical'):
    """Time-varying graphical lasso solver.

    Solves the following problem via ADMM:
        min sum_{i=1}^T -n_i log_likelihood(S_i, K_i) + alpha*||K_i||_{od,1}
            + beta sum_{i=2}^T Psi(K_i - K_{i-1})

    where S_i = (1/n_i) X_i^T \times X_i is the empirical covariance of data
    matrix X (training observations by features).

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
    init : ('empirical', 'zero')
        Choose how to initialize the precision matrix, with the inverse
        empirical covariance or zero matrix.

    Returns
    -------
    K : numpy.array, 3-dimensional (T x d x d)
        Solution to the problem for each time t=1...T .
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    psi, prox_psi, psi_node_penalty = check_norm_prox(psi)

    if init == 'empirical':
        n_times, _, n_features = emp_cov.shape
        covariance_ = emp_cov.copy()
        covariance_ *= 0.95
        K = np.empty_like(emp_cov)
        for i, (c, e) in enumerate(zip(covariance_, emp_cov)):
            c.flat[::n_features + 1] = e.flat[::n_features + 1]
            K[i] = linalg.pinvh(c)
    else:
        K = np.zeros_like(emp_cov)

    Z_0 = K.copy()  # np.zeros_like(emp_cov)
    Z_1 = K.copy()[:-1]  # np.zeros_like(emp_cov)[:-1]
    Z_2 = K.copy()[1:]  # np.zeros_like(emp_cov)[1:]
    # Z_0 = K.copy()
    # Z_1 = K.copy()[:-1]
    # Z_2 = K.copy()[1:]

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

    if n_samples is None:
        n_samples = np.ones(emp_cov.shape[0])

    checks = [
        convergence(
            obj=objective(
                n_samples, emp_cov, Z_0, K, Z_1, Z_2, alpha, beta, psi))
    ]
    for iteration_ in range(max_iter):
        # update K
        A = Z_0 - U_0
        A[:-1] += Z_1 - U_1
        A[1:] += Z_2 - U_2
        A /= divisor[:, None, None]
        # soft_thresholding_ = partial(soft_thresholding, lamda=alpha / rho)
        # K = np.array(map(soft_thresholding_, A))
        A += A.transpose(0, 2, 1)
        A /= 2.

        A *= -rho * divisor[:, None, None] / n_samples[:, None, None]
        A += emp_cov

        K = np.array(
            [
                prox_logdet(a, lamda=ni / (rho * div))
                for a, div, ni in zip(A, divisor, n_samples)
            ])

        # update Z_0
        A = K + U_0
        A += A.transpose(0, 2, 1)
        A /= 2.
        Z_0 = soft_thresholding(A, lamda=alpha / rho)

        # other Zs
        A_1 = K[:-1] + U_1
        A_2 = K[1:] + U_2
        if not psi_node_penalty:
            prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)
            Z_1 = .5 * (A_1 + A_2 - prox_e)
            Z_2 = .5 * (A_1 + A_2 + prox_e)
        else:
            Z_1, Z_2 = prox_psi(
                np.concatenate((A_1, A_2), axis=1), lamda=.5 * beta / rho,
                rho=rho, tol=tol, rtol=rtol, max_iter=max_iter)

        # update residuals
        U_0 += K - Z_0
        U_1 += K[:-1] - Z_1
        U_2 += K[1:] - Z_2

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(K - Z_0) + squared_norm(K[:-1] - Z_1) +
            squared_norm(K[1:] - Z_2))

        snorm = rho * np.sqrt(
            squared_norm(Z_0 - Z_0_old) + squared_norm(Z_1 - Z_1_old) +
            squared_norm(Z_2 - Z_2_old))

        obj = objective(
            n_samples, emp_cov, Z_0, K, Z_1, Z_2, alpha, beta, psi) \
            if compute_objective else np.nan

        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * max(
                np.sqrt(
                    squared_norm(Z_0) + squared_norm(Z_1) + squared_norm(Z_2)),
                np.sqrt(
                    squared_norm(K) + squared_norm(K[:-1]) +
                    squared_norm(K[1:]))),
            e_dual=np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * rho *
            np.sqrt(squared_norm(U_0) + squared_norm(U_1) + squared_norm(U_2)),
            # precision=Z_0.copy()
        )
        Z_0_old = Z_0.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()

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
        U_1 *= rho / rho_new
        U_2 *= rho / rho_new
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


class TimeGraphicalLasso(GraphicalLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Parameters
    ----------
    alpha : positive float, default 0.01
        Regularization parameter for precision matrix. The higher alpha,
        the more regularization, the sparser the inverse covariance.

    beta : positive float, default 1
        Regularization parameter to constrain precision matrices in time.
        The higher beta, the more regularization,
        and consecutive precision matrices in time are more similar.

    psi : {'laplacian', 'l1', 'l2', 'linf', 'node'}, default 'laplacian'
        Type of norm to enforce for consecutive precision matrices in time.

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
    covariance_ : array-like, shape (n_times, n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_times, n_features, n_features)
        Estimated precision matrix.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(
            self, alpha=0.01, beta=1., mode='admm', rho=1.,
            time_on_axis='first', tol=1e-4, rtol=1e-4, psi='laplacian',
            max_iter=100, verbose=False, assume_centered=False,
            return_history=False, update_rho_options=None,
            compute_objective=True, stop_at=None, stop_when=1e-4,
            suppress_warn_list=False):
        super(TimeGraphicalLasso, self).__init__(
            alpha=alpha, rho=rho, tol=tol, rtol=rtol, max_iter=max_iter,
            verbose=verbose, assume_centered=assume_centered, mode=mode,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective)
        self.beta = beta
        self.psi = psi
        self.time_on_axis = time_on_axis
        self.return_history = return_history
        self.stop_at = stop_at
        self.stop_when = stop_when
        self.suppress_warn_list = suppress_warn_list

    def get_observed_precision(self):
        """Getter for the observed precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.

        """
        return self.get_precision()

    def _fit(self, emp_cov, n_samples):
        """Fit the TimeGraphicalLasso model to X.

        Parameters
        ----------
        emp_cov : ndarray, shape (n_time, n_features, n_features)
            Empirical covariance of data.

        """
        out = time_graphical_lasso(
            emp_cov, alpha=self.alpha, rho=self.rho, beta=self.beta,
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

    def fit(self, X, y=None):
        """Fit the TimeGraphicalLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_time, n_samples, n_features), or
                (n_samples, n_features, n_time)
            Data from which to compute the covariance estimate.
            If shape is (n_samples, n_features, n_time), then set
            `time_on_axis = 'last'`.
        y : (ignored)

        """
        is_list = isinstance(X, list)
        X, n_samples, n_dimensions, n_times = check_input(
            X, y=None, time_on_axis=self.time_on_axis,
            suppress_warn_list=self.suppress_warn_list, estimator=self)

        if self.assume_centered:
            self.location_ = np.zeros((n_times, 1, n_dimensions))
        else:
            mean = np.array([x.mean(0) for x in X]) if is_list else X.mean(1)
            self.location_ = mean.reshape(n_times, 1, n_dimensions)
        emp_cov = np.array(
            [
                empirical_covariance(x, assume_centered=self.assume_centered)
                for x in X
            ])

        return self._fit(emp_cov, n_samples)

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
        logp : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.

        """
        X_test, n_samples, _, _ = check_input(
            X_test, y=None, time_on_axis=self.time_on_axis,
            suppress_warn_list=self.suppress_warn_list, estimator=self)

        # compute empirical covariance of the test set
        test_cov = np.array(
            [
                empirical_covariance(x, assume_centered=True)
                for x in X_test - self.location_
            ])

        logp = sum(
            n * log_likelihood(S, K) for S, K, n in zip(
                test_cov, self.get_observed_precision(), n_samples))

        # ALLA  MATLAB1
        # ranks = [np.linalg.matrix_rank(L) for L in self.latent_]
        # scores_ranks = np.square(ranks-np.sqrt(L.shape[1]))

        return logp  # - np.sum(scores_ranks)

    def error_norm(
            self, comp_cov, norm='frobenius', scaling=True, squared=True):
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
        return error_norm_time(
            self.covariance_, comp_cov, norm=norm, scaling=scaling,
            squared=squared)

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
        if self.time_on_axis == 'last':
            # put time as first dimension
            observations = observations.transpose(2, 0, 1)
        precision = self.get_observed_precision()
        # compute mahalanobis distances
        sum_ = 0.
        for obs, loc in zip(observations, self.location_):
            centered_obs = observations - self.location_
            sum_ += np.sum(np.dot(centered_obs, precision) * centered_obs, 1)

        mahalanobis_dist = sum_ / len(observations)
        return mahalanobis_dist
