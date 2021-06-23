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
from scipy import linalg
from six.moves import range
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import fast_logdet
from sklearn.utils.validation import check_array

from regain.norm import l1_od_norm
from regain.prox import prox_logdet, soft_thresholding_od
from regain.update_rules import update_rho
from regain.utils import convergence

try:
    # sklean >= 0.20
    from sklearn.covariance import GraphicalLasso as GraphLasso
except ImportError:
    # sklearn < 0.20
    from sklearn.covariance import GraphLasso


def logl(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def objective(emp_cov, x, z, alpha):
    return -logl(emp_cov, x) + l1_od_norm(alpha * z)


def init_precision(emp_cov, mode="empirical"):
    """Initialize the precision matrix given the empirical covariance."""
    if mode == "empirical":
        covariance_ = emp_cov.copy()
        covariance_ *= 0.95
        n_features = emp_cov.shape[-1]
        if emp_cov.ndim == 2:
            covariance_.flat[:: n_features + 1] = emp_cov.flat[:: n_features + 1]
            K = linalg.pinvh(covariance_)
        else:
            K = np.empty_like(emp_cov)
            for i, (c, e) in enumerate(zip(covariance_, emp_cov)):
                c.flat[:: n_features + 1] = e.flat[:: n_features + 1]
                K[i] = linalg.pinvh(c)
    elif isinstance(mode, np.ndarray):
        K = mode
    else:
        K = np.zeros_like(emp_cov)

    return K


def graphical_lasso(
    emp_cov,
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
    r"""Graphical lasso solver via ADMM.

    Solves the following problem:
        minimize  trace(S*K) - log det K + alpha ||K||_{od,1}

    where S = (1/n) X^T \times X is the empirical covariance of the data
    matrix X (training observations by features).

    Parameters
    ----------
    emp_cov : array-like
        Empirical covariance matrix.
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
    precision_ : numpy.array, 2-dimensional
        Solution to the problem.
    covariance_ : np.array, 2 dimensional
        Empirical covariance matrix.
    n_iter_ : int
        If return_n_iter, returns the number of iterations before convergence.
    history_ : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    Z = init_precision(emp_cov, mode=init)
    U = np.zeros_like(emp_cov)
    Z_old = np.zeros_like(Z)

    checks = []
    for iteration_ in range(max_iter):
        # x-update
        A = Z - U
        A += A.T
        A /= 2.0
        K = prox_logdet(emp_cov - rho * A, lamda=1.0 / rho)

        # z-update with relaxation
        K_hat = over_relax * K - (1 - over_relax) * Z
        Z = soft_thresholding_od(K_hat + U, lamda=alpha / rho)

        # update residuals
        U += K_hat - Z

        # diagnostics, reporting, termination checks
        obj = objective(emp_cov, K, Z, alpha) if compute_objective else np.nan
        rnorm = np.linalg.norm(K - Z, "fro")
        snorm = rho * np.linalg.norm(Z - Z_old, "fro")
        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=np.sqrt(K.size) * tol + rtol * max(np.linalg.norm(K, "fro"), np.linalg.norm(Z, "fro")),
            e_dual=np.sqrt(K.size) * tol + rtol * rho * np.linalg.norm(U),
        )

        Z_old = Z.copy()
        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f," "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_, **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        U *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [Z, emp_cov]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class GraphicalLasso(GraphLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

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
        super(GraphicalLasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter, verbose=verbose, assume_centered=assume_centered, mode=mode
        )
        self.rho = rho
        self.rtol = rtol
        self.over_relax = over_relax
        self.update_rho_options = update_rho_options
        self.compute_objective = compute_objective
        self.init = init

    def _fit(self, emp_cov):
        """Fit the GraphicalLasso model to X.

        Parameters
        ----------
        emp_cov : ndarray, shape (n_features, n_features)
            Empirical covariance of data.

        """
        self.precision_, self.covariance_, self.n_iter_ = graphical_lasso(
            emp_cov,
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

    def fit(self, X, y=None):
        """Fit the GraphicalLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : (ignored)

        """
        # Covariance does not make sense for a single feature
        X = check_array(X, ensure_min_features=2, ensure_min_samples=2, estimator=self)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)

        emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)
        return self._fit(emp_cov)
