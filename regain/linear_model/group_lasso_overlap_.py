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
from __future__ import division, print_function

import warnings

import numpy as np
import six
from scipy import sparse
from six.moves import range
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearClassifierMixin, LinearModel, _pre_fit
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array, check_X_y, deprecated
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

from regain.prox import soft_thresholding
from regain.utils import flatten
from regain.wrapper.paspal.glopridu import glopridu_algorithm

try:
    from regain.wrapper.paspal.wrapper import group_lasso_overlap_paspal

    _MATLAB_FOUND_ = True
except ImportError:
    _MATLAB_FOUND_ = False


def _remove_unused_features(data, groups):
    unique_idx = sorted(set(flatten(groups)))
    hashing = dict(zip(unique_idx, range(len(unique_idx))))

    new_groups = [[hashing[g] for g in group] for group in groups]
    return data[:, unique_idx], new_groups


def D_function(d, groups):
    D = np.zeros(d)
    for dimension in range(d):
        # find groups which contain this dimension
        for _, group in enumerate(groups):
            if dimension in group:
                D[dimension] += 1
    return D


def P_star_x_bar_function(x, d, groups):
    P_star_x_bar = np.zeros(d)
    for dim in range(d):
        ss = 0
        count = 0
        for g, group in enumerate(groups):
            if dim in group:
                idx = np.argwhere(np.array(group) == dim)[0]
                ss += x[g][idx]
                count += 1
        if count > 0:
            ss /= count
        P_star_x_bar[dim] = ss
    return P_star_x_bar


def group_lasso_overlap(A, b, lamda=1.0, groups=None, rho=1.0, max_iter=100, tol=1e-4, verbose=False, rtol=1e-2):
    r"""Group Lasso with Overlap solver.

    Solves the following problem via ADMM
       minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))

    The input p is a K-element vector giving the block sizes n_i, so that x_i
    is in R^{n_i}.

    Parameters
    ----------
    A : array-like, 2-dimensional
        Input matrix.
    b : array-like, 1-dimensional
        Output vector.
    lamda : float, optional
        Regularisation parameter.
    groups : list
        Groups of variables.
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
    x : numpy.array
        Solution to the problem.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    n, d = A.shape

    x = [np.zeros(len(g)) for g in groups]  # local variables
    z = np.zeros(d)
    y = [np.zeros(len(g)) for g in groups]

    D = np.diag(D_function(d, groups))
    Atb = A.T.dot(b)
    inv = np.linalg.inv(A.T.dot(A) + rho * D)
    hist = []
    count = 0
    for k in range(max_iter):
        # x update
        for i, g in enumerate(groups):
            x[i] = soft_thresholding(x[i] - y[i] / rho, lamda / rho)

        # z update
        zold = z
        x_consensus = P_star_x_bar_function(x, d, groups)
        y_consensus = P_star_x_bar_function(y, d, groups)
        z = inv.dot(Atb + D.dot(y_consensus + rho * x_consensus))

        for i, g in enumerate(groups):
            y[i] += rho * (x[i] - z[g])

        # diagnostics, reporting, termination checks
        history = (
            objective(A, b, lamda, x, z),  # objective
            np.linalg.norm(x_consensus - z),  # rnorm
            np.linalg.norm(-rho * (z - zold)),  # snorm
            np.sqrt(d) * tol + rtol * max(np.linalg.norm(x_consensus), np.linalg.norm(-z)),  # eps primal
            np.sqrt(d) * tol + rtol * np.linalg.norm(rho * y_consensus)
            # eps dual
        )

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f," "eps_pri: %.4f, eps_dual: %.4f" % history)

        hist.append(history)
        if history[1] < history[3] and history[2] < history[4]:
            if count > 10:
                break
            else:
                count += 1
        else:
            count = 0

    return z, hist, k


class GroupLassoOverlap(LinearModel, RegressorMixin):
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        groups=None,
        rho=1.0,
        n_jobs=1,
        tol=1e-4,
        verbose=False,
        rtol=1e-2,
        normalize=False,
        precompute=False,
        max_iter=1000,
        copy_X=True,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
        mode="admm",
        matlab_engine=None,
    ):
        self.alpha = alpha
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.groups = groups
        self.rho = rho
        self.verbose = verbose
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.rtol = rtol
        self.warm_start = warm_start
        self.positive = positive
        self.intercept_ = 0.0
        self.random_state = random_state
        self.selection = selection
        self.matlab_engine = matlab_engine
        self.n_jobs = n_jobs

        self.mode = mode
        if mode == "paspal-matlab" and not _MATLAB_FOUND_:
            raise ValueError("Cannot use Matlab implementation. Use `mode='admm'` or `mode='paspal'`.")

    def fit(self, X, y, check_input=True):
        """Fit model with coordinate descent.

        Parameters
        -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Notes
        -----

        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """

        if self.alpha == 0:
            warnings.warn(
                "With alpha=0, this algorithm does not converge "
                "well. You are advised to use the LinearRegression "
                "estimator",
                stacklevel=2,
            )

        if isinstance(self.precompute, six.string_types):
            raise ValueError("precompute should be one of True, False or" " array-like. Got %r" % self.precompute)

        # We expect X and y to be float64 or float32 Fortran ordered arrays
        # when bypassing checks
        if check_input:
            X, y = check_X_y(
                X,
                y,
                accept_sparse="csc",
                order="F",
                dtype=[np.float64, np.float32],
                copy=self.copy_X and self.fit_intercept,
                multi_output=True,
                y_numeric=True,
            )
            y = check_array(y, order="F", copy=False, dtype=X.dtype.type, ensure_2d=False)

        X, y, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
            X, y, None, self.precompute, self.normalize, self.fit_intercept, copy=False
        )

        if y.ndim == 1:
            y = y[:, None]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, None]

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        if self.selection not in ["cyclic", "random"]:
            raise ValueError("selection should be either random or cyclic.")

        if not self.warm_start or self.coef_ is None:
            coef_ = np.zeros((n_targets, n_features), dtype=X.dtype, order="F")
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[None, :]

        dual_gaps_ = np.zeros(n_targets, dtype=X.dtype)
        if self.n_jobs == 1:
            self.n_iter_ = []
            history = []
            for k in range(n_targets):
                if self.mode == "admm":
                    this_coef, hist, this_iter = group_lasso_overlap(
                        X,
                        y[:, k],
                        lamda=self.alpha,
                        groups=self.groups,
                        rho=self.rho,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        verbose=self.verbose,
                        rtol=self.rtol,
                    )
                elif self.mode == "paspal-matlab":
                    this_coef, hist, this_iter = group_lasso_overlap_paspal(
                        X,
                        y[:, k],
                        lamda=self.alpha,
                        groups=self.groups,
                        rho=self.rho,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        verbose=self.verbose,
                        rtol=self.rtol,
                        matlab_engine=self.matlab_engine,
                    )
                elif self.mode == "paspal":  # paspal wrapper
                    this_coef, hist, this_iter = glopridu_algorithm(
                        X,
                        y[:, k],
                        tau=self.alpha,
                        blocks=self.groups,
                        max_iter_ext=self.max_iter,
                        tol_ext=self.tol,
                        verbose=self.verbose,
                        tol_int=self.rtol,
                    )
                else:
                    raise ValueError(self.mode)
                coef_[k] = this_coef.ravel()
                history.append(hist)
                self.n_iter_.append(this_iter)
        else:
            import joblib as jl

            if self.mode == "admm":
                coef_, history, self.n_iter_ = zip(
                    *jl.Parallel(n_jobs=self.n_jobs)(
                        jl.delayed(group_lasso_overlap)(
                            X,
                            y[:, k],
                            lamda=self.alpha,
                            groups=self.groups,
                            rho=self.rho,
                            max_iter=self.max_iter,
                            tol=self.tol,
                            verbose=self.verbose,
                            rtol=self.rtol,
                        )
                        for k in range(n_targets)
                    )
                )
            elif self.mode == "paspal-matlab":  # paspal wrapper
                coef_, history, self.n_iter_ = zip(
                    *jl.Parallel(n_jobs=self.n_jobs)(
                        jl.delayed(group_lasso_overlap_paspal)(
                            X,
                            y[:, k],
                            lamda=self.alpha,
                            groups=self.groups,
                            rho=self.rho,
                            max_iter=self.max_iter,
                            tol=self.tol,
                            verbose=self.verbose,
                            rtol=self.rtol,
                            matlab_engine=self.matlab_engine,
                        )
                        for k in range(n_targets)
                    )
                )
            elif self.mode == "paspal":  # paspal wrapper
                coef_, history, self.n_iter_ = zip(
                    *jl.Parallel(n_jobs=self.n_jobs)(
                        jl.delayed(glopridu_algorithm)(
                            X,
                            y[:, k],
                            tau=self.alpha,
                            blocks=self.groups,
                            max_iter_ext=self.max_iter,
                            tol_ext=self.tol,
                            verbose=self.verbose,
                            tol_int=self.rtol,
                        )
                        for k in range(n_targets)
                    )
                )
            else:
                raise ValueError(self.mode)

        if n_targets == 1:
            self.n_iter_ = self.n_iter_[0]

        self.coef_, self.dual_gap_ = map(np.squeeze, [coef_, dual_gaps_])
        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into float64
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        self.history_ = history

        # return self for chaining fit and predict calls
        return self

    @property
    def sparse_coef_(self):
        """ sparse representation of the fitted ``coef_`` """
        return sparse.csr_matrix(self.coef_)

    @deprecated(" and will be removed in 0.19")
    def decision_function(self, X):
        """Decision function of the linear model

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        T : array, shape (n_samples,)
            The predicted decision function
        """
        return self._decision_function(X)

    def _decision_function(self, X):
        """Decision function of the linear model

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        T : array, shape (n_samples,)
            The predicted decision function
        """
        check_is_fitted(self, "n_iter_")
        if sparse.isspmatrix(X):
            return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        else:
            return super(GroupLassoOverlap, self)._decision_function(X)


class GroupLassoOverlapClassifier(LinearClassifierMixin, GroupLassoOverlap):
    """Class to extend group lasso in case of classification."""

    def fit(self, X, y, check_input=True):
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if self._label_binarizer.y_type_.startswith("multilabel"):
            # we don't (yet) support multi-label classification in ENet
            raise ValueError("%s doesn't support multi-label classification" % (self.__class__.__name__))

        # Y = column_or_1d(Y, warn=True)
        super(GroupLassoOverlapClassifier, self).fit(X, Y)
        if self.classes_.shape[0] > 2:
            ndim = self.classes_.shape[0]
        else:
            ndim = 1
        self.coef_ = self.coef_.reshape(ndim, -1)

        return self

    @property
    def classes_(self):
        return self._label_binarizer.classes_


def _overlapping_group_lasso(A, b, lamda=1.0, groups=None, rho=1.0, alpha=1.0, max_iter=100, tol=1e-4):
    # % solves the following problem via ADMM:
    # %   minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))
    # %
    # % The input p is a K-element vector giving the block sizes n_i, so that x_i
    # % is in R^{n_i}.
    # % The solution is returned in the vector x.
    # %
    # % history is a structure that contains the objective value, the primal and
    # % dual residual norms, and the tolerances for the primal and dual residual
    # % norms at each iteration.
    # % rho is the augmented Lagrangian parameter.
    # % alpha is the over-relaxation parameter (typical values for alpha are
    # % between 1.0 and 1.8).
    rtol = 1e-2

    n, d = A.shape
    N = len(groups)

    x = [np.zeros(len(g)) for g in groups]  # local variables
    z = np.zeros(d)
    y = np.zeros(d)

    hist = []
    AtA = A.T.dot(A)
    Atb = A.T.dot(b)
    inverse = np.linalg.inv(N * AtA + rho * np.eye(d))
    for k in range(max_iter):
        # % x-update (to be done in parallel)
        P_star_xk_bar = P_star_x_bar_function(x)
        for i, g in enumerate(groups):
            # x update; update each local x
            x[i] = prox(x[i] + (z - P_star_xk_bar - y / rho)[g], lamda / rho)

        P_star_xk1_bar = P_star_x_bar_function(x)
        z = inverse.dot(Atb + rho * P_star_xk1_bar + y)

        # y-update
        y += rho * (P_star_xk1_bar - z)

        # # % compute the dual residual norm square
        # s = 0
        # q = 0
        # zsold = zs
        # zs = z.reshape(-1,1).dot(np.ones((1, N)))
        # zs += Aixi - Axbar.reshape(-1,1).dot(np.ones((1, N)))
        # for i in range(N):
        #     # % dual residual norm square
        #     s = s + np.linalg.norm(-rho * Ats[i].dot((zs[:,i] - zsold[:,i])))**2
        #     # % dual residual epsilon
        #     q = q + np.linalg.norm(rho*Ats[i].dot(u))**2;
        #
        # # % diagnostics, reporting, termination checks
        # history = []
        # history.append(objective(A, b, lamda, N, x, z))
        # history.append(np.sqrt(N)*np.linalg.norm(z - Axbar))
        # history.append(np.sqrt(s))
        #
        # history.append(np.sqrt(n)*ABSTOL + rtol*max(np.linalg.norm(Aixi,'fro'), np.linalg.norm(-zs, 'fro')))
        # history.append(np.sqrt(n)*ABSTOL + rtol*np.sqrt(q))
        #
        # hist.append(history)
        # if history[1] < history[3] and history[2] < history[4]:
        #     break

    return P_star_x_bar_function(x), x


def objective(A, b, alpha, x, z):
    """Group lasso with overlap objective function."""
    return 0.5 * np.sum((A.dot(z) - b) ** 2) + alpha * np.linalg.norm(x)
