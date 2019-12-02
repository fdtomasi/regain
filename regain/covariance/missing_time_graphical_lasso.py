from __future__ import division

import warnings

import numpy as np
from six.moves import range

from sklearn.utils.validation import check_X_y

from regain.covariance.missing_graphical_lasso_ import \
        _compute_empirical_covariance, _compute_cs, _compute_mean
from regain.covariance.kernel_time_graphical_lasso_ import \
                kernel_time_graphical_lasso, KernelTimeGraphicalLasso
from regain.covariance.time_graphical_lasso_ import loss


def missing_time_graphical_lasso(
        X, alpha=0.01, rho=1,  kernel=None, psi='laplacian',
        over_relax=1, max_iter=100, verbose=False,
        tol=1e-4, rtol=1e-4, return_history=False, return_n_iter=True,
        update_rho_options=None, compute_objective=True):
    r"""Missing Graphical lasso solver via EM algorithm.

    Solves the following problem:
        minimize  trace(S*K) - log det K + alpha ||K||_{od,1}

    where S = (1/n) X^T \times X is the empirical covariance of the data
    matrix X (which contains missing data).

    Parameters
    ----------
    X : array-like shape=(n_samples, n_variables)
        Data matrix.
    alpha : float, optional
        Regularisation parameter.
    rho : float, optional
        Augmented Lagrangian parameter.
    kernel: array-like shape(n_times, n_times)
        The kernel to use to enforce similatiries among times.
    psi: string, defulat='laplacian'
        Type of consistency between networks. Option are "l1", "l2", "linf",
        "laplacian", "l12"
    over_relax : float, optional
        Over-relaxation parameter (typically between 1.0 and 1.8).
    max_iter : int, optional
        Maximum number of iterations.
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

    Returns
    -------
    X : numpy.array, 2-dimensional
        Solution to the problem.
    S : np.array, 2 dimensional
        Final empirical covariance matrix.
    n_iter : int
        If return_n_iter, returns the number of iterations before convergence.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    n_times, n_samples, d = X.shape
    K = np.zeros((n_times, d, d))
    means = np.zeros((n_times, d))

    loglik = -np.inf
    checks = []
    for iter_ in range(max_iter):
        old_logl = loglik

        cs = np.array([_compute_cs(means[t, :], K[t, :, :], X[t, :, :])
                       for t in range(n_times)])
        means = np.array([_compute_mean(X[t, :, :], cs[t, :, :])
                          for t in range(n_times)])
        emp_cov = np.array([
                    _compute_empirical_covariance(X[t, :, :], K[t, :, :],
                                                  cs[t, :, :])
                    for t in range(n_times)
                    ])
        print(emp_cov)
        K = kernel_time_graphical_lasso(
                emp_cov, alpha=alpha, rho=rho, kernel=kernel,
                max_iter=max_iter, verbose=max(0, verbose-1),
                psi=psi, tol=tol, rtol=tol,
                return_history=False, return_n_iter=True, mode='admm',
                update_rho_options=None, compute_objective=False, stop_at=None,
                stop_when=1e-4, init='empirical')[0]

        loglik = loss(emp_cov, K)
        diff = old_logl - loglik
        checks.append(dict(iteration=iter_,
                           log_likelihood=loglik,
                           difference=diff))
        if verbose:
            print("Iter %d: log-likelihood %.4f, difference: %.4f" % (
                    iter_, loglik, diff))
        if iter_ > 1 and diff < tol:
            break
    else:
        warnings.warn("The Missing Graphical Lasso algorithm did not converge")
    aux = np.nan_to_num(np.copy(X))
    aux += cs
    return_list = [K, emp_cov, aux]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iter_)
    return return_list


class MissingTimeGraphicalLasso(KernelTimeGraphicalLasso):
    """Time-Varying Graphical Lasso with missing data.

    This method allows for graphical model selection in presence of missing
    data in the dataset. It is suitable to perform imputing after fitting.

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

    update_rho_options : dict, default None
        Options for the update of rho. See `update_rho` function for details.

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
            self, alpha=0.01, kernel=None, rho=1., tol=1e-4, rtol=1e-4,
            psi='laplacian', max_iter=100, verbose=False,
            return_history=False,
            update_rho_options=None, compute_objective=True, ker_param=1,
            max_iter_ext=100):
        super(MissingTimeGraphicalLasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter, verbose=verbose,
            assume_centered=False, rho=rho,
            rtol=rtol, kernel=kernel, psi=psi,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective)

    def fit(self, X, y):
        """Fit the MissingTimeGraphicalLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : ndarray, shape (n_samples, 1)
            Division in times.

        """
        X, y = check_X_y(
                X, y, accept_sparse=False, dtype=np.float64, order="C",
                ensure_min_features=2, estimator=self,
                force_all_finite='allow-nan')
        self.classes_, n_samples = np.unique(y, return_counts=True)
        X = np.array([X[y == cl] for cl in self.classes_])
        self.precision_, self.covariance_, self.complete_data_matrix_, \
            self.n_iter_ = missing_time_graphical_lasso(
                X, alpha=self.alpha, tol=self.tol,
                max_iter=self.max_iter,
                verbose=self.verbose, rho=self.rho,
                rtol=self.rtol, beta=self.beta, kernel=self.kernel,
                psi=self.psi, return_n_iter=True,
                update_rho_options=self.update_rho_options,
                compute_objective=self.compute_objective)
        return self
