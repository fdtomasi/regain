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
"""Graphical latent variable model selection via ADMM."""
from __future__ import division

import warnings

import numpy as np
from scipy import linalg
from six.moves import range

from regain.covariance.graphical_lasso_ import GraphicalLasso, init_precision
from regain.covariance.graphical_lasso_ import objective as obj_gl
from regain.prox import prox_logdet, prox_trace_indicator, soft_thresholding
from regain.update_rules import update_rho
from regain.utils import convergence


def objective(emp_cov, R, K, L, alpha, tau):
    """Objective function for latent graphical lasso."""
    obj = obj_gl(emp_cov, R, K, alpha=alpha)
    obj += tau * np.linalg.norm(L, ord="nuc")
    return obj


def latent_graphical_lasso(
    emp_cov,
    alpha=1.0,
    tau=1.0,
    rho=1.0,
    max_iter=100,
    verbose=False,
    tol=1e-4,
    rtol=1e-2,
    return_history=False,
    return_n_iter=True,
    update_rho_options=None,
    compute_objective=True,
    init="empirical",
):
    r"""Latent variable graphical lasso solver via ADMM.

    Solves the following problem:
        min - log_likelihood(S, K-L) + alpha ||K||_{od,1} + tau ||L_i||_*

    where S = (1/n) X^T \times X is the empirical covariance of the data
    matrix X (training observations by features).

    Parameters
    ----------
    emp_cov : array-like
        Empirical covariance matrix.
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
    K, L : np.array, 2-dimensional, size (d x d)
        Solution to the problem.
    S : np.array, 2 dimensional
        Empirical covariance matrix.
    n_iter : int
        If return_n_iter, returns the number of iterations before convergence.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    K = init_precision(emp_cov, mode=init)
    L = np.zeros_like(emp_cov)
    U = np.zeros_like(emp_cov)
    R_old = np.zeros_like(emp_cov)

    checks = []
    for iteration_ in range(max_iter):
        # update R
        A = K - L - U
        A += A.T
        A /= 2.0
        R = prox_logdet(emp_cov - rho * A, lamda=1.0 / rho)

        A = L + R + U
        K = soft_thresholding(A, lamda=alpha / rho)

        A = K - R - U
        A += A.T
        A /= 2.0
        L = prox_trace_indicator(A, lamda=tau / rho)

        # update residuals
        U += R - K + L

        # diagnostics, reporting, termination checks
        obj = objective(emp_cov, R, K, L, alpha, tau) if compute_objective else np.nan
        rnorm = np.linalg.norm(R - K + L)
        snorm = rho * np.linalg.norm(R - R_old)
        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=np.sqrt(R.size) * tol + rtol * max(np.linalg.norm(R), np.linalg.norm(K - L)),
            e_dual=np.sqrt(R.size) * tol + rtol * rho * np.linalg.norm(U),
        )
        R_old = R.copy()

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f," "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
        if check.obj == np.inf:
            break
        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_, **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        U *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    covariance_ = linalg.pinvh(K)
    return_list = [K, L, covariance_]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class LatentGraphicalLasso(GraphicalLasso):
    """Sparse inverse covariance + low-rank matrix estimation.

    Parameters
    ----------
    alpha : positive float, default 0.01
        Regularization parameter for precision matrix. The higher alpha,
        the more regularization, the sparser the inverse covariance.

    tau : positive float, default 1
        Regularization parameter for latent variables matrix. The higher tau,
        the more regularization, the lower rank of the latent matrix.

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

    init : {'empirical', 'zeros', ndarray}, default 'empirical'
        How to initialise the inverse covariance matrix. Default is take
        the empirical covariance and inverting it.

    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    latent_ : array-like, shape (n_features, n_features)
        Estimated latent variable matrix.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(
        self,
        alpha=0.01,
        tau=1.0,
        rho=1.0,
        tol=1e-4,
        rtol=1e-4,
        max_iter=100,
        verbose=False,
        assume_centered=False,
        mode="admm",
        update_rho_options=None,
        compute_objective=True,
        init="empirical",
    ):
        super(LatentGraphicalLasso, self).__init__(
            alpha=alpha,
            rho=rho,
            tol=tol,
            rtol=rtol,
            max_iter=max_iter,
            verbose=verbose,
            assume_centered=assume_centered,
            mode=mode,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective,
            init=init,
        )
        self.tau = tau

    def get_precision(self):
        """Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.
            Note that this is the observed precision matrix.

        """
        return self.precision_ - self.latent_

    def _fit(self, emp_cov):
        """Fit the LatentGraphicalLasso model to X.

        Parameters
        ----------
        emp_cov : ndarray, shape (n_features, n_features)
            Empirical covariance of data.

        """
        self.precision_, self.latent_, self.covariance_, self.n_iter_ = latent_graphical_lasso(
            emp_cov,
            alpha=self.alpha,
            tau=self.tau,
            rho=self.rho,
            tol=self.tol,
            rtol=self.rtol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            return_n_iter=True,
            return_history=False,
            update_rho_options=self.update_rho_options,
            compute_objective=self.compute_objective,
            init=self.init,
        )
        return self
