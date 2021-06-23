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
"""Sparse inverse covariance selection via ADMM.

More information can be found in the paper linked at:
http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
"""
from __future__ import division

import warnings

import numpy as np
from six.moves import range

from regain.covariance.graphical_lasso_ import GraphicalLasso, graphical_lasso, logl


def compute_empirical_covariance(X, K, cs):
    emp_cov = np.zeros((X.shape[0], K.shape[0], K.shape[0]))
    aux = np.nan_to_num(np.copy(X))
    aux += cs
    for i in range(X.shape[0]):
        for v in range(emp_cov.shape[1]):
            for s in range(emp_cov.shape[1]):
                if np.isnan(X[i, v]) and np.isnan(X[i, s]):
                    nans = np.where(np.isnan(X[i, :]))[0]
                    xxm, yym = np.meshgrid(nans, nans)
                    inv = np.linalg.pinv(K[xxm, yym])[np.where(nans == v)[0][0], np.where(nans == s)[0][0]]
                    emp_cov[i, v, s] = inv + cs[i, v] * cs[i, s]
                else:
                    emp_cov[i, v, s] = aux[i, v] * aux[i, s]
    emp_cov = np.sum(emp_cov, axis=0)
    return emp_cov / np.max(emp_cov)


def compute_cs(means, K, X):
    cs = np.zeros_like(X)
    for i in range(X.shape[0]):
        nans = np.where(np.isnan(X[i, :]))[0]
        obs = np.where(np.logical_not(np.isnan(X[i, :])))[0]
        xxm, yym = np.meshgrid(nans, nans)
        xxm1, yyo = np.meshgrid(obs, nans)
        KK = np.linalg.pinv(K[xxm, yym]).dot(K[xxm1, yyo])
        cs[i, nans] = means[nans] - KK.dot(X[i, obs].T - means[obs])
    return cs / max(np.max(np.abs(cs)), 1)


def compute_mean(X, cs):
    aux = np.nan_to_num(np.copy(X))
    aux += cs
    return np.sum(aux, axis=0)


def missing_graphical_lasso(
    X,
    alpha=0.01,
    rho=1,
    over_relax=1,
    max_iter=100,
    verbose=False,
    tol=1e-4,
    rtol=1e-4,
    return_history=False,
    return_n_iter=True,
    update_rho_options=None,
    compute_objective=True,
    init="empirical",
):
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
    init : {'empirical', 'zeros', ndarray}, default 'empirical'
        How to initialise the inverse covariance matrix. Default is take
        the empirical covariance and inverting it.

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
    K = np.eye(X.shape[1])
    means = np.zeros(X.shape[1])

    loglik = -np.inf
    checks = []
    for iter_ in range(max_iter):
        old_logl = loglik

        cs = compute_cs(means, K, X)
        means = compute_mean(X, cs)
        emp_cov = compute_empirical_covariance(X, K, cs)
        K, _ = graphical_lasso(
            emp_cov,
            alpha=alpha,
            rho=rho,
            over_relax=over_relax,
            max_iter=max_iter,
            verbose=max(0, int(verbose - 1)),
            tol=tol,
            rtol=rtol,
            return_history=False,
            return_n_iter=False,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective,
            init=K,
        )
        loglik = logl(emp_cov, K)
        diff = old_logl - loglik
        checks.append(dict(iteration=iter_, log_likelihood=logl, difference=diff))
        if verbose:
            print("Iter %d: log-likelihood %.4f, difference: %.4f" % (iter_, loglik, diff))
        if np.abs(diff) < tol:
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


class MissingGraphicalLasso(GraphicalLasso):
    """Graphical Lasso with missing data.

    This method allows for graphical model selection in presence of missing
    data in the dataset. It is suitable to perform imputing after fitting.

    Parameters
    ----------
    alpha : positive float, default 0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

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
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(
        self,
        alpha=0.01,
        rho=1.0,
        over_relax=1.0,
        max_iter=100,
        mode="admm",
        tol=1e-4,
        rtol=1e-4,
        verbose=False,
        assume_centered=False,
        update_rho_options=None,
        compute_objective=True,
        init="empirical",
    ):
        super(MissingGraphicalLasso, self).__init__(
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            assume_centered=assume_centered,
            mode=mode,
            rho=rho,
            rtol=rtol,
            over_relax=over_relax,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective,
            init=init,
        )

    def fit(self, X, y=None):
        """Fit the GraphicalLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : (ignored)

        """
        # Covariance does not make sense for a single feature
        # X = check_array(
        #     X, ensure_min_features=2, ensure_min_samples=2, estimator=self)

        self.precision_, self.covariance_, self.complete_data_matrix_, self.n_iter_ = missing_graphical_lasso(
            X,
            alpha=self.alpha,
            tol=self.tol,
            rtol=self.rtol,
            max_iter=self.max_iter,
            over_relax=self.over_relax,
            rho=self.rho,
            verbose=self.verbose,
            return_n_iter=True,
            return_history=False,
            update_rho_options=self.update_rho_options,
            compute_objective=self.compute_objective,
            init=self.init,
        )
        return self
