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
import warnings

import numpy as np

from sklearn.utils import check_array
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from regain.generalized_linear_model.base import GLM_GM, convergence
from regain.generalized_linear_model.base import build_adjacency_matrix
from regain.prox import soft_thresholding_od
from regain.norm import l1_od_norm


def loss(X, theta):
    n, d = X.shape
    objective = 0
    if not np.all(theta == theta.T):
        return np.float("inf")
    for r in range(d):
        selector = [i for i in range(d) if i != r]
        for i in range(n):
            XXT = X[i, r] * X[i, selector].dot(theta[selector, r])
            XT = X[i, selector].dot(theta[selector, r])
            EXT = np.exp(XT)
            E_XT = np.exp(-XT)
            objective += np.log(EXT + E_XT) - XXT
    return (1 / n) * objective


def objective(X, theta, alpha):
    n, _ = X.shape
    objective = loss(X, theta)
    return objective + alpha * l1_od_norm(theta)


def _gradient_ising(X, theta, n, A=None, rho=1, T=0):
    n, d = X.shape
    theta_new = np.zeros_like(theta)

    def gradient(X, thetas, r, selector, n, A=None, rho=1, T=0):
        sum_ = np.zeros((1, len(selector)))
        for i in range(X.shape[0]):
            XT = X[i, selector].dot(theta[selector, r])
            EXT = np.exp(XT)
            E_XT = np.exp(-XT)
            sum_ += (1 / n) * X[i, selector] * ((EXT - E_XT) / (EXT + E_XT) - X[i, r])
            return sum_

    for ix in range(theta.shape[0]):
        selector = [i for i in range(d) if i != ix]
        theta_new[ix, selector] = gradient(X, theta, ix, selector, n, A, rho, T)
    if A is not None:
        theta_new += (rho * T) * (theta - A)

    return theta_new


def _fit(
    X,
    alpha=1e-2,
    gamma=1e-3,
    tol=1e-3,
    max_iter=1000,
    verbose=0,
    return_history=True,
    compute_objective=True,
    warm_start=None,
    return_n_iter=False,
    adjust_gamma=False,
    A=None,
    T=0,
    rho=1,
    update_gamma=0.5,
    line_search=False,
):
    n, d = X.shape
    if warm_start is None:
        theta = np.zeros((d, d))
    else:
        theta = check_array(warm_start)

    thetas = [theta]
    theta_new = theta.copy()
    checks = []
    for iter_ in range(max_iter):
        theta_old = thetas[-1]
        if not line_search:
            grad = _gradient_ising(X, theta, n, A, rho, T)
            theta_new = theta - gamma * grad
            theta = (theta_new + theta_new.T) / 2
            theta = soft_thresholding_od(theta, alpha * gamma)
        else:
            while True:
                grad = _gradient_ising(X, theta, n, A, rho, T)
                theta_new = theta - gamma * grad
                theta = (theta_new + theta_new.T) / 2
                theta = soft_thresholding_od(theta, alpha * gamma)
                print(theta)
                loss_new = loss(X, theta)
                loss_old = loss(X, theta_old)
                # Line search
                diff_theta2 = np.linalg.norm(theta_old - theta) ** 2
                grad_diff = np.trace(grad.dot(theta_old - theta))
                diff = loss_old - grad_diff + (diff_theta2 / (2 * gamma))

                if loss_new > diff or np.isinf(loss_new) or np.isnan(loss_new):
                    gamma = update_gamma * gamma
                    theta = theta_old - gamma * grad
                    theta = soft_thresholding_od(theta, alpha * gamma)
                    loss_new = loss(X, theta)
                    diff = loss_old - grad_diff + (diff_theta2 / (2 * gamma))
                else:
                    break
        thetas.append(theta)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check = convergence(
                iter=iter_,
                obj=objective(X, theta, alpha),
                iter_norm=np.linalg.norm(thetas[-2] - thetas[-1]),
                iter_r_norm=(np.linalg.norm(thetas[-2] - thetas[-1]) / np.linalg.norm(thetas[-1])),
            )
        checks.append(check)
        # if adjust_gamma: # TODO multiply or divide
        if verbose:
            print("Iter: %d, objective: %.4f, iter_norm %.4f" % (check[0], check[1], check[2]))

        if np.abs(check[2]) < tol:
            break

    return_list = [thetas[-1]]
    if return_history:
        return_list.append(thetas)
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iter_)

    return return_list


class IsingGraphicalModel(GLM_GM, BaseEstimator):
    """Graphical model inference with Bernoulli distribution.

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
        mode="symmetric_fbs",
        rho=1,
        max_iter=100,
        verbose=False,
        return_history=True,
        return_n_iter=False,
        compute_objective=True,
        gamma=1,
    ):
        super(IsingGraphicalModel, self).__init__(
            alpha, tol, rtol, max_iter, verbose, return_history, return_n_iter, compute_objective
        )
        self.reconstruction = reconstruction
        self.mode = mode
        self.rho = rho
        self.gamma = gamma

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
            res = _fit(X, self.alpha, tol=self.tol, gamma=self.gamma, max_iter=self.max_iter, verbose=self.verbose)
            self.precision_ = res[0]
            self.history = res[1:]
        elif self.mode.lower() == "coordinate_descent":
            raise ValueError("Not implemented")
            # thetas_pred = []
            # historys = []
            # for ix in range(X.shape[1]):
            #     res = fit_each_variable(X, ix, self.alpha, tol=self.tol,
            #                             gamma=self.gamma,
            #                             verbose=self.verbose)
            #     thetas_pred.append(res[0])
            #     historys.append(res[1:])
            # self.precision_ = build_adjacency_matrix(thetas_pred,
            #                                          how=self.reconstruction)
            # self.history = historys
        elif self.mode.lower() == "logistic_regression":
            thetas_pred = []
            for ix in range(X.shape[1]):
                verbose = min(0, self.verbose - 1)
                selector = np.array([i for i in range(X.shape[1]) if i != ix])
                print("pd")
                res = (
                    LogisticRegression(
                        C=1 / self.alpha, penalty="l1", solver="liblinear", verbose=verbose, random_state=0
                    )
                    .fit(X[:, selector], X[:, ix])
                    .coef_
                )
                thetas_pred.append(res)
            self.precision_ = build_adjacency_matrix(thetas_pred, how=self.reconstruction)
        else:
            raise ValueError(
                "Unknown optimization mode. Found " + self.mode + ". Options are 'coordiante_descent', "
                "'symmetric_fbs'"
            )
        return self

    def score(self, X, y=None):
        return 0
