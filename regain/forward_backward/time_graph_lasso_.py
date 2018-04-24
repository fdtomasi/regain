"""Time graph lasso via forward_backward (for now only in case of l1 norm)."""
from __future__ import division, print_function

import warnings
from functools import partial

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from six.moves import map, range, zip
from sklearn.covariance import empirical_covariance, log_likelihood
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_array

from regain.covariance.time_graph_lasso_ import TimeGraphLasso, loss
from regain.norm import l1_od_norm, vector_p_norm
from regain.prox import prox_FL
# from regain.update_rules import update_gamma
from regain.utils import convergence, positive_definite
from regain.validation import check_array_dimensions


def penalty(precision, alpha, beta, psi):
    """Penalty for time-varying graphical lasso."""
    obj = alpha * sum(map(l1_od_norm, precision))
    obj += beta * sum(map(psi, precision[1:] - precision[:-1]))
    return obj


def objective(n_samples, emp_cov, precision, alpha, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = loss(emp_cov, precision, n_samples=n_samples)
    obj += penalty(precision, alpha, beta, psi)
    return obj


def grad_loss(x, emp_cov, n_samples, x_inv=None):
    """Gradient of the loss function for the time-varying graphical lasso."""
    if x_inv is None:
        x_inv = np.array([linalg.pinvh(_) for _ in x])
    grad = emp_cov - x_inv
    return grad * n_samples[:, None, None]


def _J(x, beta, alpha, gamma, lamda, S, n_samples, p=1, x_inv=None, grad=None):
    """Grad + prox + line search for the new point."""
    if grad is None:
        grad = grad_loss(x, S, n_samples, x_inv=x_inv)
    prox = prox_FL(x - gamma * grad, beta * gamma, alpha * gamma, p=p)
    return x + lamda * (prox - x)


def _scalar_product(x, y):
    return np.hstack(x).dot(np.hstack(y).T).sum()


def choose_gamma(gamma, x, emp_cov, n_samples, beta, alpha, lamda, grad,
                 delta=1e-4, eps=0.5, max_iter=1000, p=1, x_inv=None):
    """Choose alpha for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741

    """
    if grad is None:
        grad = grad_loss(x, emp_cov, n_samples, x_inv=x_inv)

    partial_f = partial(loss, n_samples=n_samples, S=emp_cov)
    fx = partial_f(K=x)
    for i in range(max_iter):
        prox = prox_FL(x - gamma * grad, beta * gamma, alpha * gamma, p=p)
        y_minus_x = prox - x
        loss_diff = partial_f(K=x + lamda * y_minus_x) - fx

        tolerance = _scalar_product(y_minus_x, grad)
        tolerance += delta / gamma * _scalar_product(y_minus_x, y_minus_x)
        if loss_diff <= lamda * tolerance:
            return gamma
        gamma *= eps
    return gamma


def choose_lamda(lamda, x, emp_cov, n_samples, beta, alpha, gamma, delta=1e-4,
                 eps=0.5, max_iter=1000, criterion='b', p=1, x_inv=None,
                 grad=None, prox=None):
    """Choose alpha for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741

    """
    # lamda = 1.
    if x_inv is None:
        x_inv = np.array([linalg.pinvh(_) for _ in x])
    if grad is None:
        grad = grad_loss(x, emp_cov, n_samples, x_inv=x_inv)
    if prox is None:
        prox = _J(x, beta=beta, alpha=alpha, lamda=1, gamma=gamma, S=emp_cov,
                  n_samples=n_samples, p=p, x_inv=x_inv, grad=grad)

    partial_f = partial(loss, n_samples=n_samples, S=emp_cov)
    fx = partial_f(K=x)

    y_minus_x = prox - x
    if criterion == 'b':
        tolerance = _scalar_product(y_minus_x, grad)
        tolerance += delta / gamma * _scalar_product(y_minus_x, y_minus_x)
    elif criterion == 'c':
        gx = penalty(x, alpha, beta, partial(vector_p_norm, p=p))
        gy = penalty(prox, alpha, beta, partial(vector_p_norm, p=p))
        objective_x = objective(
            n_samples, emp_cov, x, alpha, beta, partial(vector_p_norm, p=p))
        tolerance = (1 - delta) * (gy - gx + _scalar_product(y_minus_x, grad))

    for i in range(max_iter):
        # line-search
        x1 = x + lamda * y_minus_x

        if criterion == 'a':
            iter_diff = x1 - x
            # iter_diff_gradient = np.hstack(iter_diff).dot(np.hstack(grad).T).sum()
            gradx1 = grad_loss(x1, emp_cov, n_samples)
            grad_diff = gradx1 - grad
            # if np.linalg.norm(grad_diff) <= tolerance:
            norm_grad_diff = np.sqrt(_scalar_product(grad_diff, grad_diff))
            norm_iter_diff = np.sqrt(_scalar_product(iter_diff, iter_diff))
            tolerance = delta * norm_iter_diff / (gamma * lamda)
            if norm_grad_diff <= tolerance:
                return lamda
        elif criterion == 'b':
            loss_diff = partial_f(K=x1) - fx
            # tolerance = delta * squared_norm(iter_diff) / (gamma * lamda)
            # if loss_diff - iter_diff_gradient <= tolerance:
            #     return lamda

            # after some mathematical reductions ...
            if loss_diff <= lamda * tolerance:
                return lamda
        elif criterion == 'c':
            obj_diff = objective(
                n_samples, emp_cov, x1, alpha, beta,
                partial(vector_p_norm, p=p)) - objective_x

            if obj_diff <= lamda * tolerance:
                return lamda
        else:
            raise ValueError(criterion)
        lamda *= eps
    return lamda


def fista_step(Y, Y_diff, t):
    t_next = (1. + np.sqrt(1.0 + 4.0 * t*t)) / 2.
    return Y + ((t - 1.0)/t_next) * Y_diff, t_next


def upper_diag_3d(x):
    """Return the flattened upper diagonal of a 3d matrix."""
    n_times, n_dim, _ = x.shape
    upper_idx = np.triu_indices(n_dim, 1)
    return np.array([xx[upper_idx] for xx in x])


def time_graph_lasso(
        emp_cov, n_samples, alpha=0.01, beta=1., max_iter=100, verbose=False,
        tol=1e-4, delta=1e-4, gamma=1., lamda=1., eps=0.5,
        return_history=False, return_n_iter=True, choose='gamma',
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
    if choose not in ('gamma', 'lamda', 'fixed'):
        raise ValueError("`choose` parameter must be one of %s." % (
            ('gamma', 'lamda', 'fixed'),))

    n_times, _, n_features = emp_cov.shape
    covariance_ = emp_cov.copy()
    covariance_ *= 0.95

    K = np.empty_like(emp_cov)
    for i, (c, e) in enumerate(zip(covariance_, emp_cov)):
        c.flat[::n_features + 1] = e.flat[::n_features + 1]
        K[i] = linalg.pinvh(c)

    # K = np.array([np.eye(s.shape[0]) for s in emp_cov])
    # Y = K.copy()

    checks = []
    obj_partial = partial(
        objective, n_samples=n_samples, emp_cov=emp_cov,
        alpha=alpha, beta=beta, psi=partial(vector_p_norm, p=time_norm))
    max_residual = -np.inf
    for iteration_ in range(max_iter):
        if not positive_definite(K):
            warnings.warn("precision is not positive definite.")
            break

        k_previous = K.copy()

        # choose a gamma
        x_inv = np.array([linalg.pinvh(x) for x in K])

        # total variation
        # Y = _J(K, beta, alpha, gamma, 1, S, n_samples)
        grad = grad_loss(K, emp_cov, n_samples, x_inv=x_inv)
        if choose == 'gamma':
            gamma = choose_gamma(
                gamma / eps, K, emp_cov, n_samples=n_samples,
                beta=beta, alpha=alpha, lamda=lamda, grad=grad,
                delta=delta, eps=eps, max_iter=200, p=time_norm, x_inv=x_inv)
        # else:
        #     gamma = update_gamma(gamma, iteration_, eps=1e-4)

        x_hat = K - gamma * grad
        y = prox_FL(x_hat, beta * gamma, alpha * gamma, p=time_norm)

        if choose == 'lamda':
            lamda = choose_lamda(
                lamda / eps, K, emp_cov, n_samples=n_samples,
                beta=beta, alpha=alpha,
                gamma=gamma, delta=delta, eps=eps,
                criterion=lamda_criterion, max_iter=200, p=time_norm,
                x_inv=x_inv, grad=grad, prox=y)

        K = K + np.maximum(lamda, 0) * (y - K)
        # K = K + choose_lamda(lamda, K, emp_cov, n_samples, beta, alpha,
        #                      gamma, delta=delta, criterion=lamda_criterion,
        #                      max_iter=50) * (Y - K)

        # K, t = fista_step(Y, Y - Y_old, t)

        check = convergence(
            obj=obj_partial(precision=K),
            rnorm=np.linalg.norm(upper_diag_3d(K) - upper_diag_3d(k_previous)),
            snorm=np.linalg.norm(
                obj_partial(precision=K) - obj_partial(precision=k_previous)),
            e_pri=np.sqrt(upper_diag_3d(K).size) * tol + tol * max(
                np.linalg.norm(upper_diag_3d(K)),
                np.linalg.norm(upper_diag_3d(k_previous))),
            e_dual=tol)

        if verbose and iteration_ % (50 if verbose < 2 else 1) == 0:
            print("obj: %.4f, rnorm: %.7f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        if return_history:
            checks.append(check)

        if np.isnan(check.rnorm) or np.isnan(check.snorm):
            warnings.warn("precision is not positive definite.")

        # use this convergence criterion
        subgrad = (x_hat - K) / gamma
        if 1:
            grad = grad_loss(K, emp_cov, n_samples)
            # grad = upper_diag_3d(grad)
            # subgrad = upper_diag_3d(subgrad)
            res_norm = np.linalg.norm(grad + subgrad)

            if iteration_ == 0:
                normalizer = res_norm + 1e-6
            max_residual = max(np.linalg.norm(grad),
                               np.linalg.norm(subgrad)) + 1e-6
        else:
            res_norm = np.linalg.norm(K - k_previous) / gamma
            max_residual = max(max_residual, res_norm)
            normalizer = max(np.linalg.norm(grad),
                             np.linalg.norm(subgrad)) + 1e-6

        r_rel = res_norm / max_residual
        r_norm = res_norm / normalizer
        # print(r_rel, r_norm)

        if (r_rel <= tol or r_norm <= tol): # or (
                # check.rnorm <= check.e_pri and iteration_ > 0):
            break
        # if check.rnorm <= check.e_pri and iteration_ > 0:
        #     # and check.snorm <= check.e_dual:
        #     break
    else:
        warnings.warn("Objective did not converge.")

    for i in range(K.shape[0]):
        covariance_[i] = linalg.pinvh(K[i])

    return_list = [K, covariance_]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_ + 1)
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
                 compute_objective=True, eps=0.5, choose='gamma', lamda=1,
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
        self.choose = choose
        self.lamda = lamda

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
                gamma=self.gamma, delta=self.delta, eps=self.eps,
                choose=self.choose, lamda=self.lamda)
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
            self.alpha = self.alpha_max(emp_cov, is_covariance=True)

        # if self.gamma == 'max':
        #     lipschitz_constant = max(get_lipschitz(e) for e in emp_cov)
        #     self.gamma = 1.98 / lipschitz_constant

        return self._fit(emp_cov, n_samples)

    def alpha_max(self, X, is_covariance=False):
        """Compute the alpha_max for the problem at hand, based on sklearn."""
        from sklearn.covariance.graph_lasso_ import alpha_max
        if is_covariance:
            emp_cov = X
        else:
            emp_cov = np.array([empirical_covariance(
                x, assume_centered=self.assume_centered) for x in X])
        return max(alpha_max(e) for e in emp_cov)

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
        res = sum(n * log_likelihood(S, K) for S, K, n in zip(
            test_cov, self.get_observed_precision(), n_samples))

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
