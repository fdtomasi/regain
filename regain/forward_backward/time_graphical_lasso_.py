# BSD 3-Clause License

# Copyright (c) 2019, regain authors
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
"""Time graph lasso via forward_backward (for now only in case of l1 norm)."""
from __future__ import division, print_function

import warnings
from functools import partial

import numpy as np
from scipy import linalg
from six.moves import map, range, zip
from sklearn.covariance.graph_lasso_ import alpha_max
from sklearn.utils.extmath import squared_norm

from regain.covariance.time_graphical_lasso_ import TimeGraphicalLasso, init_precision
from regain.covariance.time_graphical_lasso_ import loss as loss_tgl
from regain.norm import l1_od_norm, vector_p_norm
from regain.prox import prox_FL, soft_thresholding_od
from regain.utils import convergence

from .forward_backward import _scalar_product, choose_gamma, choose_lamda, upper_diag_3d


def loss(S, K, n_samples=None, vareps=0):
    """Loss function for time-varying graphical lasso."""
    loss_ = loss_tgl(S, K, n_samples=n_samples)
    loss_ += vareps / 2.0 * _scalar_product(K, K)
    return loss_


def grad_loss(x, emp_cov, n_samples, x_inv=None, vareps=0):
    """Gradient of the loss function for the time-varying graphical lasso."""
    if x_inv is None:
        x_inv = np.array([linalg.pinvh(_) for _ in x])
    grad = emp_cov - x_inv
    grad *= n_samples[:, None, None]

    # add coercitive term
    grad += vareps * x

    return grad


def penalty(precision, alpha, beta, psi):
    """Penalty for time-varying graphical lasso."""
    if isinstance(alpha, np.ndarray):
        obj = sum(a[0][0] * m for a, m in zip(alpha, map(l1_od_norm, precision)))
    else:
        obj = alpha * sum(map(l1_od_norm, precision))
    if isinstance(beta, np.ndarray):
        obj += sum(b[0][0] * m for b, m in zip(beta, map(psi, precision[1:] - precision[:-1])))
    else:
        obj += beta * psi(precision[1:] - precision[:-1])
    return obj


def objective(n_samples, emp_cov, precision, alpha, beta, psi, vareps=0):
    """Objective function for time-varying graphical lasso."""
    obj = loss(emp_cov, precision, n_samples=n_samples, vareps=vareps)
    obj += penalty(precision, alpha, beta, psi)
    return obj


def loss_laplacian(S, K, beta=0, n_samples=None, vareps=0):
    """Loss function for TGL with laplacian penalty."""
    loss_ = loss(S, K, n_samples=n_samples, vareps=vareps)
    loss_ += beta * squared_norm(K[1:] - K[:-1])
    return loss_


def grad_loss_laplacian(x, emp_cov, beta=0, n_samples=None, x_inv=None, vareps=0):
    """Gradient of the loss function for TGL with laplacian penalty."""
    grad = grad_loss(x, emp_cov, n_samples, x_inv, vareps)

    aux = np.empty_like(x)
    aux[0] = x[0] - x[1]
    aux[-1] = x[-1] - x[-2]
    for t in range(1, x.shape[0] - 1):
        aux[t] = 2 * x[t] - x[t - 1] - x[t + 1]
    aux *= 2 * beta
    grad += aux

    return grad


def penalty_laplacian(precision, alpha):
    """Laplacian penalty for time-varying graphical lasso."""
    if isinstance(alpha, np.ndarray):
        obj = sum(a[0][0] * m for a, m in zip(alpha, map(l1_od_norm, precision)))
    else:
        obj = alpha * sum(map(l1_od_norm, precision))
    # obj += beta * psi(precision[1:] - precision[:-1])
    return obj


def objective_laplacian(n_samples, emp_cov, precision, alpha, beta, vareps=0):
    """Objective function for TGL with Laplacian penalty."""
    obj = loss_laplacian(emp_cov, precision, beta=beta, n_samples=n_samples, vareps=vareps)
    obj += penalty_laplacian(precision, alpha)
    return obj


def _J(x, beta, alpha, gamma, lamda, S, n_samples, p=1, x_inv=None, grad=None):
    """Grad + prox + line search for the new point."""
    prox = prox_FL(x - gamma * grad, beta * gamma, alpha * gamma, p=p, symmetric=True)
    return x + lamda * (prox - x)


def tgl_forward_backward(
    emp_cov,
    alpha=0.01,
    beta=1.0,
    max_iter=100,
    n_samples=None,
    verbose=False,
    tol=1e-4,
    delta=1e-4,
    gamma=1.0,
    lamda=1.0,
    eps=0.5,
    debug=False,
    return_history=False,
    return_n_iter=True,
    choose="gamma",
    lamda_criterion="b",
    time_norm=1,
    compute_objective=True,
    return_n_linesearch=False,
    vareps=1e-5,
    stop_at=None,
    stop_when=1e-4,
    laplacian_penalty=False,
    init="empirical",
):
    """Time-varying graphical lasso solver with forward-backward splitting.

    Solves the following problem via FBS:
        min sum_{i=1}^T -n_i log_likelihood(S_i, K_i) + alpha*||K_i||_{od,1}
            + beta sum_{i=2}^T Psi(K_i - K_{i-1})

    where S_i = (1/n_i) X_i^T \times X_i is the empirical covariance of data
    matrix X (training observations by features).

    Parameters
    ----------
    emp_cov : ndarray, shape (n_times, n_features, n_features)
        Empirical covariance of data.
    alpha, beta : float, optional
        Regularisation parameters.
    max_iter : int, optional
        Maximum number of iterations.
    n_samples : ndarray
        Number of samples available for each time point.
    verbose : bool, default False
        Print info at each iteration.
    tol : float, optional
        Absolute tolerance for convergence.
    delta, gamma, lamda, eps : float, optional
        FBS parameters.
    debug : bool, default False
        Run in debug mode.
    return_history : bool, optional
        Return the history of computed values.
    return_n_iter : bool, optional
        Return the number of iteration before convergence.
    choose : ('gamma', 'lambda', 'fixed', 'both)
        Search iteratively gamma / lambda / none / both.
    lamda_criterion : ('a', 'b', 'c')
        Criterion to choose lamda. See ref for details.
    time_norm : float, optional
        Choose the temporal norm between points.
    compute_objective : bool, default True
        Choose to compute the objective value.
    return_n_linesearch : bool, optional
        Return the number of line-search iterations before convergence.
    vareps : float, optional
        Jitter for the loss.
    stop_at, stop_when : float, optional
        Other convergence criteria, as used in the paper.
    laplacian_penalty : bool, default False
        Use Laplacian penalty.
    init : {'empirical', 'zero', ndarray}
        Choose how to initialize the precision matrix, with the inverse
        empirical covariance, zero matrix or precomputed.

    Returns
    -------
    K, covariance : numpy.array, 3-dimensional (T x d x d)
        Solution to the problem for each time t=1...T .
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    available_choose = ("gamma", "lamda", "fixed", "both")
    if choose not in available_choose:
        raise ValueError("`choose` parameter must be one of %s." % available_choose)

    n_times, _, n_features = emp_cov.shape
    K = init_precision(emp_cov, mode=init)

    if laplacian_penalty:
        obj_partial = partial(
            objective_laplacian, n_samples=n_samples, emp_cov=emp_cov, alpha=alpha, beta=beta, vareps=vareps
        )
        function_f = partial(loss_laplacian, beta=beta, n_samples=n_samples, S=emp_cov, vareps=vareps)
        gradient_f = partial(grad_loss_laplacian, emp_cov=emp_cov, beta=beta, n_samples=n_samples, vareps=vareps)
        function_g = partial(penalty_laplacian, alpha=alpha)
    else:
        psi = partial(vector_p_norm, p=time_norm)
        obj_partial = partial(
            objective, n_samples=n_samples, emp_cov=emp_cov, alpha=alpha, beta=beta, psi=psi, vareps=vareps
        )
        function_f = partial(loss, n_samples=n_samples, S=emp_cov, vareps=vareps)
        gradient_f = partial(grad_loss, emp_cov=emp_cov, n_samples=n_samples, vareps=vareps)
        function_g = partial(penalty, alpha=alpha, beta=beta, psi=psi)

    max_residual = -np.inf
    n_linesearch = 0
    checks = [convergence(obj=obj_partial(precision=K))]
    for iteration_ in range(max_iter):
        k_previous = K.copy()
        x_inv = np.array([linalg.pinvh(x) for x in K])
        grad = gradient_f(K, x_inv=x_inv)

        if choose in ["gamma", "both"]:
            gamma, y = choose_gamma(
                gamma / eps if iteration_ > 0 else gamma,
                K,
                function_f=function_f,
                beta=beta,
                alpha=alpha,
                lamda=lamda,
                grad=grad,
                delta=delta,
                eps=eps,
                max_iter=200,
                p=time_norm,
                x_inv=x_inv,
                choose=choose,
                laplacian_penalty=laplacian_penalty,
            )

        x_hat = K - gamma * grad
        if choose not in ["gamma", "both"]:
            if laplacian_penalty:
                y = soft_thresholding_od(x_hat, alpha * gamma)
            else:
                y = prox_FL(x_hat, beta * gamma, alpha * gamma, p=time_norm, symmetric=True)

        if choose in ("lamda", "both"):
            lamda, n_ls = choose_lamda(
                min(lamda / eps if iteration_ > 0 else lamda, 1),
                K,
                function_f=function_f,
                objective_f=obj_partial,
                gradient_f=gradient_f,
                function_g=function_g,
                gamma=gamma,
                delta=delta,
                eps=eps,
                criterion=lamda_criterion,
                max_iter=200,
                p=time_norm,
                grad=grad,
                prox=y,
                vareps=vareps,
            )
            n_linesearch += n_ls

        K = K + min(max(lamda, 0), 1) * (y - K)
        # K, t = fista_step(Y, Y - Y_old, t)

        check = convergence(
            obj=obj_partial(precision=K),
            rnorm=np.linalg.norm(upper_diag_3d(K) - upper_diag_3d(k_previous)),
            snorm=np.linalg.norm(obj_partial(precision=K) - obj_partial(precision=k_previous)),
            e_pri=np.sqrt(upper_diag_3d(K).size) * tol
            + tol * max(np.linalg.norm(upper_diag_3d(K)), np.linalg.norm(upper_diag_3d(k_previous))),
            e_dual=tol,
        )

        if verbose and iteration_ % (50 if verbose < 2 else 1) == 0:
            print("obj: %.4f, rnorm: %.7f, snorm: %.4f," "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        if return_history:
            checks.append(check)

        if np.isnan(check.rnorm) or np.isnan(check.snorm):
            warnings.warn("precision is not positive definite.")

        if stop_at is not None:
            if abs(check.obj - stop_at) / abs(stop_at) < stop_when:
                break
        else:
            # use this convergence criterion
            subgrad = (x_hat - K) / gamma
            if 0:
                if laplacian_penalty:
                    grad = grad_loss_laplacian(K, emp_cov, n_samples, vareps=vareps)
                else:
                    grad = grad_loss(K, emp_cov, n_samples, vareps=vareps)
                res_norm = np.linalg.norm(grad + subgrad)

                if iteration_ == 0:
                    normalizer = res_norm + 1e-6
                max_residual = max(np.linalg.norm(grad), np.linalg.norm(subgrad)) + 1e-6
            else:
                res_norm = np.linalg.norm(K - k_previous) / gamma
                max_residual = max(max_residual, res_norm)
                normalizer = max(np.linalg.norm(grad), np.linalg.norm(subgrad)) + 1e-6

            r_rel = res_norm / max_residual
            r_norm = res_norm / normalizer

            if not debug and (r_rel <= tol or r_norm <= tol) and iteration_ > 0:  # or (
                # check.rnorm <= check.e_pri and iteration_ > 0):
                break
    else:
        warnings.warn("Objective did not converge.")

    covariance_ = np.array([linalg.pinvh(k) for k in K])
    return_list = [K, covariance_]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_ + 1)
    if return_n_linesearch:
        return_list.append(n_linesearch)
    return return_list


class TimeGraphicalLassoForwardBackward(TimeGraphicalLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Parameters
    ----------
    alpha, beta : float, optional
        Regularisation parameters.

    max_iter : int, optional
        Maximum number of iterations.

    n_samples : ndarray
        Number of samples available for each time point.

    verbose : bool, default False
        Print info at each iteration.

    tol : float, optional
        Absolute tolerance for convergence.

    delta, gamma, lamda, eps : float, optional
        FBS parameters.

    debug : bool, default False
        Run in debug mode.

    return_history : bool, optional
        Return the history of computed values.

    return_n_iter : bool, optional
        Return the number of iteration before convergence.

    choose : ('gamma', 'lambda', 'fixed', 'both)
        Search iteratively gamma / lambda / none / both.

    lamda_criterion : ('a', 'b', 'c')
        Criterion to choose lamda. See ref for details.

    time_norm : float, optional
        Choose the temporal norm between points.

    compute_objective : bool, default True
        Choose to compute the objective value.

    return_n_linesearch : bool, optional
        Return the number of line-search iterations before convergence.

    vareps : float, optional
        Jitter for the loss.

    stop_at, stop_when : float, optional
        Set a budget for the iterations.

    laplacian_penalty : bool, default False
        Use Laplacian penalty.

    init : {'empirical', 'zero', ndarray}
        Choose how to initialize the precision matrix, with the inverse
        empirical covariance, zero matrix or precomputed.

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
        self,
        alpha=0.01,
        beta=1.0,
        tol=1e-4,
        max_iter=100,
        verbose=False,
        assume_centered=False,
        compute_objective=True,
        eps=0.5,
        choose="gamma",
        lamda=1,
        delta=1e-4,
        gamma=1.0,
        lamda_criterion="b",
        time_norm=1,
        return_history=False,
        debug=False,
        return_n_linesearch=False,
        vareps=1e-5,
        stop_at=None,
        stop_when=1e-4,
        init="empirical",
        laplacian_penalty=False,
    ):
        super(TimeGraphicalLassoForwardBackward, self).__init__(
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            assume_centered=assume_centered,
            compute_objective=compute_objective,
            beta=beta,
            init=init,
        )
        self.delta = delta
        self.gamma = gamma
        self.lamda_criterion = lamda_criterion
        self.time_norm = time_norm
        self.eps = eps
        self.choose = choose
        self.lamda = lamda
        self.return_history = return_history
        self.debug = debug
        self.return_n_linesearch = return_n_linesearch
        self.vareps = vareps
        self.stop_at = stop_at
        self.stop_when = stop_when
        self.laplacian_penalty = laplacian_penalty

    def _fit(self, emp_cov, n_samples):
        """Fit the TimeGraphLasso model to X.

        Parameters
        ----------
        emp_cov : ndarray, shape (n_time, n_features, n_features)
            Empirical covariance of data.

        """
        if self.alpha == "max":
            self.alpha = max(alpha_max(e) for e in emp_cov)

        out = tgl_forward_backward(
            emp_cov,
            n_samples=n_samples,
            alpha=self.alpha,
            beta=self.beta,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            return_n_iter=True,
            return_history=self.return_history,
            compute_objective=self.compute_objective,
            time_norm=self.time_norm,
            lamda_criterion=self.lamda_criterion,
            gamma=self.gamma,
            delta=self.delta,
            eps=self.eps,
            choose=self.choose,
            lamda=self.lamda,
            debug=self.debug,
            return_n_linesearch=self.return_n_linesearch,
            vareps=self.vareps,
            stop_at=self.stop_at,
            stop_when=self.stop_when,
            init=self.init,
            laplacian_penalty=self.laplacian_penalty,
        )

        if self.return_history:
            if self.return_n_linesearch:
                self.precision_, self.covariance_, self.history_, self.n_iter_, self.n_linesearch_ = out
            else:
                self.precision_, self.covariance_, self.history_, self.n_iter_ = out
        else:
            if self.return_n_linesearch:
                self.precision_, self.covariance_, self.n_iter_, self.n_linesearch_ = out
            else:
                self.precision_, self.covariance_, self.n_iter_ = out
        return self
