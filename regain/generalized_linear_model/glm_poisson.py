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

import numpy as np
from sklearn.utils import check_array
from sklearn.base import BaseEstimator

from regain.generalized_linear_model.base import GLM_GM, convergence
from regain.generalized_linear_model.base import build_adjacency_matrix
from regain.prox import soft_thresholding
from regain.norm import l1_od_norm


def loss_single_variable(X, theta, n, r, selector):
    objective = 0
    for i in range(n):
        XXT = X[i, r] * X[i, selector].dot(theta)
        expXT = np.exp(X[i, selector].dot(theta))
        objective += expXT - XXT
    return (1 / n) * objective


def objective(X, theta, alpha):
    n, d = X.shape
    objective = 0
    if not np.all(theta == theta.T):
        return np.float("inf")
    for r in range(d):
        selector = [i for i in range(d) if i != r]
        objective += objective_single_variable(X, theta[r, selector], n, r, selector, 0)
    return objective + alpha * l1_od_norm(theta)


def objective_single_variable(X, theta, n, r, selector, alpha):
    objective = 0
    for i in range(X.shape[0]):
        XXT = X[i, r] * X[i, selector].dot(theta)
        expXT = np.exp(X[i, selector].dot(theta))
        objective += expXT - XXT
    return (1 / n) * objective + alpha * np.linalg.norm(theta, 1)


def fit_each_variable(
    X,
    ix,
    alpha=1e-2,
    gamma=1,
    tol=1e-3,
    max_iter=100,
    verbose=0,
    update_gamma=0.5,
    return_history=True,
    compute_objective=True,
    return_n_iter=False,
    adjust_gamma=False,
    A=None,
    T=0,
    rho=1,
):
    n, d = X.shape
    theta = np.zeros(d - 1)
    selector = [i for i in range(d) if i != ix]

    def gradient(X, theta, r, selector, n, A, T, rho):
        XTX = X[:, selector].T.dot(X[:, r])
        EXK = X[:, selector].T.dot(np.exp(X[:, selector].dot(theta)))
        to_add = 0
        if A is not None:
            to_add = (rho * T) * (theta - A[r, selector]) / n
        return -(1 / n) * (XTX - EXK) + to_add

    thetas = [theta]
    checks = []
    for iter_ in range(max_iter):
        theta_old = thetas[-1]
        grad = gradient(X, theta, ix, selector, n, A, T, rho)
        while True:
            theta_new = theta - gamma * grad
            theta = soft_thresholding(theta_new, alpha * gamma)
            loss_new = loss_single_variable(X, theta, n, ix, selector)
            loss_old = loss_single_variable(X, theta_old, n, ix, selector)
            # Line search
            diff_theta2 = np.linalg.norm(theta_old - theta) ** 2
            grad_diff = grad.dot(theta_old - theta)
            diff = loss_old - grad_diff + (diff_theta2 / (2 * gamma))

            if loss_new > diff or np.isinf(loss_new) or np.isnan(loss_new):
                gamma = update_gamma * gamma
                theta = theta_old - gamma * grad
                theta = soft_thresholding(theta, alpha * gamma)
                loss_new = loss_single_variable(X, theta, n, ix, selector)
                diff = loss_old - grad_diff + (diff_theta2 / (2 * gamma))
            else:
                break
        thetas.append(theta)
        if iter_ > 0:
            check = convergence(
                iter=iter_,
                obj=objective_single_variable(X, theta, n, ix, selector, alpha),
                iter_norm=np.linalg.norm(thetas[-2] - thetas[-1]),
                iter_r_norm=(np.linalg.norm(thetas[-2] - thetas[-1]) / np.linalg.norm(thetas[-2])),
            )
            checks.append(check)
            # if adjust_gamma: # TODO multiply or divide
            if verbose:
                print(
                    "Iter: %d, objective: %.4f, iter_norm %.4f,"
                    " iter_norm_normalized: %.4f" % (check[0], check[1], check[2], check[3])
                )

            if np.abs(check[2]) < tol:
                break

    return_list = [thetas[-1]]
    if return_history:
        return_list.append(thetas)
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iter_)

    return return_list


def loss(X, theta):
    n, d = X.shape
    objective = 0
    for r in range(d):
        selector = [i for i in range(d) if i != r]
        a = loss_single_variable(X, theta[r, selector], n, r, selector)
        objective += a
    return objective


class PoissonGraphicalModel(GLM_GM, BaseEstimator):
    """Graphical model inference with local Poisson distribution.

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

    compute_objective : boolean, default True
        Choose if compute the objective function during iterations
        (only useful if `verbose=True`).

    n_cores: int, default -1
         Number of cores to use in parallel execution.

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
        tol=1e-4,
        rtol=1e-4,
        reconstruction="union",
        mode="coordinate_descent",
        max_iter=100,
        gamma=0.1,
        intercept=False,
        verbose=False,
        return_history=True,
        return_n_iter=False,
        compute_objective=True,
    ):
        super(PoissonGraphicalModel, self).__init__(
            alpha, tol, rtol, max_iter, verbose, return_history, return_n_iter, compute_objective
        )
        self.reconstruction = reconstruction
        self.mode = mode
        self.gamma = gamma
        self.intercept = intercept

    def get_precision(self):
        return self.precision_

    def fit(self, X, y=None):
        """
        X : ndarray, shape = (n_samples * n_times, n_dimensions)
            Data matrix.
        y : added for compatiblity
        gamma: float,
            Step size of the proximal gradient descent.
        """
        X = check_array(X)
        if self.mode.lower() == "symmetric_fbs":
            raise ValueError("Not implemented.")

        elif self.mode.lower() == "coordinate_descent":
            print("sono qui")
            thetas_pred = []
            historys = []
            if self.intercept:
                X = np.hstack((X, np.ones((X.shape[0], 1))))
            for ix in range(X.shape[1]):
                verbose = max(0, self.verbose - 1)
                res = fit_each_variable(X, ix, self.alpha, tol=self.tol, verbose=verbose)
                thetas_pred.append(res[0])
                historys.append(res[1:])
            self.precision_ = build_adjacency_matrix(thetas_pred, how=self.reconstruction)
            self.history = historys
        else:
            raise ValueError(
                "Unknown optimization mode. Found " + self.mode + ". Options are 'coordiante_descent', "
                "'symmetric_fbs'"
            )
        return self

    def score(self, X, y=None):
        return 0
