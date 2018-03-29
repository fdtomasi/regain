"""Time graph lasso via forward_backward (for now only in case of l1 norm)."""
from __future__ import division

import warnings
from functools import partial

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from six.moves import map, range, zip
from sklearn.covariance import empirical_covariance, log_likelihood
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_array

from regain.covariance.graph_lasso_ import fast_logdet
from regain.covariance.time_graph_lasso_ import TimeGraphLasso, loss
from regain.norm import l1_od_norm, vector_p_norm
from regain.prox import prox_FL
from regain.update_rules import update_gamma
from regain.utils import convergence
from regain.validation import check_array_dimensions


def penalty(precision, alpha, beta, psi):
    obj = alpha * sum(map(l1_od_norm, precision))
    obj += beta * sum(map(psi, precision[1:] - precision[:-1]))
    return obj


def objective(n_samples, emp_cov, precision, alpha, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = loss(emp_cov, precision, n_samples=n_samples)
    obj += penalty(precision, alpha, beta, psi)
    return obj


def grad_loss(x, emp_cov, n_samples):
    """Gradient of the loss function for the time-varying graphical lasso."""
    grad = emp_cov - np.array([linalg.pinvh(_) for _ in x])
    return grad * n_samples[:, None, None]


def _J(x, beta, alpha, gamma, lamda, S, n_samples, p=1):
    """Grad + prox + line search for the new point."""
    grad = grad_loss(x, S, n_samples)
    prox = prox_FL(x - gamma * grad, beta * gamma, alpha * gamma, p=p)
    return x + lamda * (prox - x)


def choose_lamda(lamda, x, emp_cov, n_samples, beta, alpha, gamma, delta=1e-4,
                 eps=0.5, max_iter=1000, criterion='a', p=1):
    """Choose alpha for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741
    """
    # lamda = 1.
    partial_j = partial(_J, x, beta=beta, alpha=alpha, gamma=gamma, S=emp_cov,
                        n_samples=n_samples, p=p)
    partial_f = partial(loss, n_samples=n_samples, S=emp_cov)
    fx = partial_f(K=x)
    gradx = grad_loss(x, emp_cov, n_samples)
    gx = penalty(x, lamda, beta, partial(vector_p_norm, p=p))
    for i in range(max_iter):
        x1 = partial_j(lamda=lamda)
        iter_diff = x1 - x
        loss_diff = partial_f(K=x1) - fx
        iter_diff_gradient = iter_diff.ravel().dot(gradx.ravel())

        if criterion == 'a':
            tolerance = delta * np.linalg.norm(iter_diff) / (gamma * lamda)
            gradx1 = grad_loss(x1, emp_cov, n_samples)
            grad_diff = gradx1.ravel() - gradx.ravel()
            if np.linalg.norm(grad_diff) <= tolerance:
                # print("Choose lamda = %.2f" % lamda)
                return lamda
        elif criterion == 'b':
            tolerance = delta * squared_norm(iter_diff) / (gamma * lamda)
            if loss_diff - iter_diff_gradient <= tolerance:
                # print("Choose lamda = %.2f" % lamda)
                return lamda
        elif criterion == 'c':
            obj_diff = objective(
                n_samples, emp_cov, x1, lamda, beta,
                partial(vector_p_norm, p=p)) - \
                objective(n_samples, emp_cov, x, lamda,
                          beta, partial(vector_p_norm, p=p))
            y = _J(x, beta, alpha, gamma, 1, emp_cov, n_samples, p=p)
            gy = penalty(y, lamda, beta, partial(vector_p_norm, p=p))
            tolerance = (1 - delta) * lamda * (
                gy - gx + (y - x).ravel().dot(gradx.ravel()))
            if obj_diff <= tolerance:
                # print("Choose lamda = %.2f" % lamda)
                return lamda
        else:
            raise ValueError(criterion)
        lamda *= eps
    return lamda


def fista_step(Y, Y_diff, t):
    t_next = (1. + np.sqrt(1.0 + 4.0 * t*t)) / 2.
    return Y + ((t - 1.0)/t_next) * Y_diff, t_next


def time_graph_lasso(
        emp_cov, n_samples, alpha=0.01, beta=1., max_iter=100, verbose=False,
        tol=1e-4, delta=1e-4, gamma=1., eps=0.5,
        return_history=False, return_n_iter=True,
        lamda_criterion='b', time_norm=1, compute_objective=True):
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
    n_times, _, n_features = emp_cov.shape
    covariance_ = emp_cov.copy()
    covariance_ *= 0.95

    K = []
    for c, e in zip(covariance_, emp_cov):
        c.flat[::n_features + 1] = e.flat[::n_features + 1]
        K.append(linalg.pinvh(c))

    # K = np.array([np.eye(s.shape[0]) for s in emp_cov])
    K = np.array(K)
    # Y = K.copy()

    checks = []
    lamda = 1
    # t = 1
    obj_partial = partial(
        objective, n_samples=n_samples, emp_cov=emp_cov,
        alpha=alpha, beta=beta, psi=partial(vector_p_norm, p=time_norm))
    for iteration_ in range(max_iter):
        k_previous = K.copy()  # np.ones_like(S) + 5000
        # Y_old = Y.copy()

        # choose a gamma
        gamma = update_gamma(gamma, iteration_, eps=1e-4)

        # total variation
        # Y = _J(K, beta, alpha, gamma, 1, S, n_samples)
        y = prox_FL(K - gamma * grad_loss(K, emp_cov, n_samples),
                    beta * gamma, alpha * gamma, p=time_norm)

        lamda_n = choose_lamda(
            lamda, K, emp_cov, n_samples=n_samples, beta=beta, alpha=alpha,
            gamma=gamma, delta=delta, eps=eps,
            criterion=lamda_criterion, max_iter=40, p=time_norm)

        K += np.maximum(lamda_n, 1e-3) * (y - K)
        # K = K + choose_lamda(lamda, K, emp_cov, n_samples, beta, alpha,
        #                      gamma, delta=delta, criterion=lamda_criterion,
        #                      max_iter=50) * (Y - K)

        # K, t = fista_step(Y, Y - Y_old, t)

        check = convergence(
            obj=obj_partial(precision=K),
            rnorm=np.linalg.norm(K - k_previous),
            snorm=np.linalg.norm(
                obj_partial(precision=K) - obj_partial(precision=k_previous)),
            e_pri=tol, e_dual=tol)

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)
            # print("K: %s" % K)
            # print("Kold: %s" % k_previous)

        if return_history:
            checks.append(check)

        if np.isnan(check.rnorm) or np.isnan(check.snorm):
            # raise ValueError("%f %f" % (check.rnorm, check.snorm))
            warnings.warn("precision is not positive definite.")

        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
    else:
        warnings.warn("Objective did not converge.")

    return_list = [K, emp_cov]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class TimeGraphLassoForwardBackward(TimeGraphLasso):
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
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(self, alpha=0.01, beta=1., time_on_axis='first', tol=1e-4,
                 max_iter=100, verbose=False, assume_centered=False,
                 compute_objective=True, eps=0.5,
                 delta=1e-4, gamma=1., lamda_criterion='b', time_norm=1):
        super(TimeGraphLassoForwardBackward, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            verbose=verbose, assume_centered=assume_centered,
            compute_objective=compute_objective, beta=beta,
            time_on_axis=time_on_axis)
        self.delta = delta
        self.gamma = gamma
        self.lamda_criterion = lamda_criterion
        self.time_norm = time_norm
        self.eps = eps

    def _fit(self, emp_cov, n_samples):
        """Fit the TimeGraphLasso model to X.

        Parameters
        ----------
        emp_cov : ndarray, shape (n_time, n_features, n_features)
            Empirical covariance of data.

        """
        self.precision_, self.covariance_, self.n_iter_ = \
            time_graph_lasso(
                emp_cov, n_samples=n_samples, alpha=self.alpha, beta=self.beta,
                tol=self.tol, max_iter=self.max_iter, verbose=self.verbose,
                return_n_iter=True, return_history=False,
                compute_objective=self.compute_objective,
                time_norm=self.time_norm, lamda_criterion=self.lamda_criterion,
                gamma=self.gamma, delta=self.delta, eps=self.eps)
        return self

    def fit(self, X, y=None):
        """Fit the TimeGraphLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_time, n_samples, n_features), or
                (n_samples, n_features, n_time)
            Data from which to compute the covariance estimate.
            If shape is (n_samples, n_features, n_time), then set
            `time_on_axis = 'last'`.
        y : (ignored)

        """
        if sp.issparse(X):
            raise TypeError("sparse matrices not supported.")

        X = check_array_dimensions(
            X, n_dimensions=3, time_on_axis=self.time_on_axis)

        # Covariance does not make sense for a single feature
        X = np.array([check_array(x, ensure_min_features=2,
                      ensure_min_samples=2, estimator=self) for x in X])

        if self.assume_centered:
            self.location_ = np.zeros((X.shape[0], 1, X.shape[2]))
        else:
            self.location_ = X.mean(1).reshape(X.shape[0], 1, X.shape[2])
        emp_cov = np.array([empirical_covariance(
            x, assume_centered=self.assume_centered) for x in X])
        n_samples = np.array([x.shape[0] for x in X])

        if self.alpha == 'max':
            # use sklearn alpha max
            from sklearn.covariance.graph_lasso_ import alpha_max
            self.alpha = max(alpha_max(e) for e in emp_cov) + 0.4
        if self.gamma == 'max':
            lipschitz_constant = max(get_lipschitz(e) for e in emp_cov)
            self.gamma = 1.98 / lipschitz_constant

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
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.

        """
        if sp.issparse(X_test):
            raise TypeError("sparse matrices not supported.")

        X_test = check_array_dimensions(
            X_test, n_dimensions=3, time_on_axis=self.time_on_axis)

        # Covariance does not make sense for a single feature
        X_test = np.array([
            check_array(x, ensure_min_features=2,
                        ensure_min_samples=2, estimator=self) for x in X_test])

        # compute empirical covariance of the test set
        test_cov = np.array([empirical_covariance(
            x, assume_centered=True) for x in X_test - self.location_])

        n_samples = np.array([x.shape[0] for x in X_test])
        res = sum(log_likelihood(S, K) for S, K in zip(
            test_cov, self.get_observed_precision()))

        return res


def get_lipschitz(data):
    """Get the Lipschitz constant for a specific loss function.

    Only square loss implemented.

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    loss : string
        the selected loss function in {'square', 'logit'}
    Returns
    ----------
    L : float
        the Lipschitz constant
    """
    n, p = data.shape

    if p > n:
        tmp = np.dot(data, data.T)
    else:
        tmp = np.dot(data.T, data)
    return np.linalg.norm(tmp, 2)
