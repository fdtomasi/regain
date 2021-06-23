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


from regain.generalized_linear_model.base import GLM_GM, convergence, build_adjacency_matrix
from regain.prox import soft_thresholding


def objective(X, theta, n, r, selector, alpha):
    XXT = X[:, r].T.dot(X[:, selector]).dot(theta)
    TXXT = theta.T.dot(X[:, selector].T).dot(X[:, selector]).dot(theta)
    return -(1 / n) * XXT + (1 / (2 * n)) * TXXT + alpha * np.linalg.norm(theta, 1)


def fit_each_variable(
    X,
    ix,
    alpha=1e-2,
    gamma=1e-3,
    tol=1e-3,
    max_iter=1000,
    verbose=0,
    return_history=True,
    compute_objective=True,
    return_n_iter=False,
    adjust_gamma=False,
):
    n, d = X.shape
    theta = np.zeros(d - 1) + 1e-15
    selector = [i for i in range(d) if i != ix]

    def gradient(X, theta, r, selector, n):
        XX = X[:, r].T.dot(X[:, selector])
        XXT = X[:, selector].T.dot(X[:, selector]).dot(theta)
        return -(1 / n) * XX + (1 / n) * XXT

    thetas = [theta]
    checks = []
    for iter_ in range(max_iter):
        theta_new = theta - gamma * gradient(X, theta, ix, selector, n)
        theta = soft_thresholding(theta_new, alpha * gamma)
        thetas.append(theta)

        check = convergence(
            iter=iter_,
            obj=objective(X, theta, n, ix, selector, alpha),
            iter_norm=np.linalg.norm(thetas[-2] - thetas[-1]),
            iter_r_norm=(np.linalg.norm(thetas[-2] - thetas[-1]) / np.linalg.norm(thetas[-1])),
        )
        checks.append(check)
        # if adjust_gamma: # TODO multiply or divide
        if verbose:
            print("Iter: %d, objective: %.4f, iter_norm %.4f" % (check[0], check[1], check[2]))

        if check[-2] < tol:
            break

    return_list = [thetas[-1]]
    if return_history:
        return_list.append(thetas)
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iter_)

    return return_list


class Gaussian_GLM_GM(GLM_GM):
    """Graphical model inference with Gaussian distribution.

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
        max_iter=100,
        verbose=False,
        return_history=True,
        return_n_iter=False,
        compute_objective=True,
    ):
        super(Gaussian_GLM_GM, self).__init__(
            alpha, tol, rtol, max_iter, verbose, return_history, return_n_iter, compute_objective
        )
        self.reconstruction = reconstruction

    def get_precision(self):
        return self.precision_

    def fit(self, X, y=None, gamma=1e-3):
        """
        X : ndarray, shape = (n_samples * n_times, n_dimensions)
            Data matrix.
        y : added for compatiblity
        gamma: float,
            Step size of the proximal gradient descent.
        """
        X = check_array(X)
        thetas_pred = []
        historys = []
        for ix in range(X.shape[1]):
            res = fit_each_variable(X, ix, self.alpha)
            thetas_pred.append(res[0])
            historys.append(res[1:])
        self.precision_ = build_adjacency_matrix(thetas_pred, how=self.reconstruction)
        self.history = historys
        return self
