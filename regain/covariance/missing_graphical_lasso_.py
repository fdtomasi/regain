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

from __future__ import division

import warnings

import numpy as np
from six.moves import range
from scipy.linalg import pinvh

from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import fast_logdet
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_random_state, check_array

from regain.covariance.graphical_lasso_ import graphical_lasso
from regain.covariance.graphical_lasso_ import GraphicalLasso, logl
from regain.utils import convergence


def _compute_empirical_covariance(X, K, cs):
    emp_cov = np.zeros((X.shape[0], K.shape[0], K.shape[0]))
    aux = np.nan_to_num(np.copy(X))
    aux += cs
    for i in range(X.shape[0]):
        for v in range(emp_cov.shape[1]):
            for s in range(emp_cov.shape[1]):
                if np.isnan(X[i, v]) and np.isnan(X[i, s]):
                    nans = np.where(np.isnan(X[i, :]))[0]
                    xxm, yym = np.meshgrid(nans, nans)
                    inv = np.linalg.pinv(K[xxm, yym])[
                        np.where(nans == v)[0][0], np.where(nans == s)[0][0]]
                    emp_cov[i, v, s] = inv + cs[i, v]*cs[i, s]
                else:
                    emp_cov[i, v, s] = aux[i, v]*aux[i, s]
    emp_cov = np.sum(emp_cov, axis=0)
    return emp_cov/np.max(emp_cov)


def _compute_cs(means, K, X):
    cs = np.zeros_like(X)
    for i in range(X.shape[0]):
        nans = np.where(np.isnan(X[i, :]))[0]
        obs = np.where(np.logical_not(np.isnan(X[i, :])))[0]
        xxm, yym = np.meshgrid(nans, nans)
        xxm1, yyo = np.meshgrid(obs, nans)
        KK = np.linalg.pinv(K[xxm, yym]).dot(K[xxm1, yyo])
        cs[i, nans] = means[nans] - KK.dot(X[i, obs].T - means[obs])
    return cs/max(np.max(np.abs(cs)), 1)


def _compute_mean(X, cs):
    aux = np.nan_to_num(np.copy(X))
    aux += cs
    return np.sum(aux, axis=0)


def missing_graphical_lasso(
        X, alpha=0.01, rho=1, over_relax=1, max_iter=100, verbose=False,
        tol=1e-4, rtol=1e-4, return_history=False, return_n_iter=True,
        update_rho_options=None, compute_objective=True, init='empirical'):
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
    K = np.zeros((X.shape[1], X.shape[1]))
    means = np.zeros(X.shape[1])

    loglik = -np.inf
    checks = []
    for iter_ in range(max_iter):
        old_logl = loglik

        cs = _compute_cs(means, K, X)
        means = _compute_mean(X, cs)
        emp_cov = _compute_empirical_covariance(X, K, cs)
        K, _ = graphical_lasso(emp_cov, alpha=alpha, rho=rho,
                               over_relax=over_relax, max_iter=max_iter,
                               verbose=max(0, int(verbose-1)),
                               tol=tol*10, rtol=rtol*10, return_history=False,
                               return_n_iter=False,
                               update_rho_options=update_rho_options,
                               compute_objective=compute_objective,
                               init=K)
        loglik = logl(emp_cov, K)
        diff = old_logl - loglik
        checks.append(dict(iteration=iter_,
                           log_likelihood=logl,
                           difference=diff))
        if verbose:
            print("Iter %d: log-likelihood %.4f, difference: %.4f" % (
                    iter_, loglik, diff))
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


def _penalized_nll(K, S=None, regularizer=None):
    res = - fast_logdet(K) + np.sum(K*S)
    res += np.linalg.norm(regularizer*K, 1)
    return res


def latent_missing_graphical_lasso(
        emp_cov, M, alpha, mu, eta=0, rho=1,
        tol=1e-3, max_iter=200, verbose=0, compute_objective=False,
        return_n_iter=False, penalize_latent=True, max_iter_graph_lasso=100):
    """Graphical Lasso with missing data as latent variables.

    This method allows for graphical model selection in presence of missing
    data in the dataset. It is suitable to estimate latent variables samples.
    For references see:
    "Yuan, Ming. Discussion: Latent variable graphical model selection via
    convex optimization. Ann. Statist. 40 (2012), no. 4, 1968--1972."

    "Tozzo, Veronica, et al. "Group induced graphical lasso allows for
    discovery of molecular pathways-pathways interactions." arXiv preprint
    arXiv:1811.09673 (2018)."

    Parameters
    ----------
    emp_cov: ndarray, shape (n_features, n_features)
        Empirical covariance of data.

    M: array-like, shape=(n_dim_obs, n_dim_lat)
        Prior knowledge to put on the connections between latent and observed
        variables. If mask is a matrix of zeros the algorithm corresponds to
        the one of Yuan et al.(2012), in the other case to Tozzo et al.(2018).

    alpha : positive float, default 0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

    mu :  positive float, default 0.01
        The regularization parameter on the inter-links: the higher mu, the
        more the final matrix will have links similar to the ones in mask.

    eta :  positive float, default 0.1
        The regularization parameter on the latent variables: the higher eta,
        the sparser the network on the latent variables.

    rho : positive float, default 1
        Augmented Lagrangian parameter.

    tol : positive float, default 1e-4
        Absolute tolerance to declare convergence.

    max_iter : integer, default 100
        The maximum number of iterations.

    verbose : boolean, default False
        If verbose is True, the objective function, rnorm and snorm are
        printed at each iteration.

    compute_objective : boolean, default True
        Choose if compute the objective function during iterations
        (only useful if `verbose=True`).

    return_n_iter : boolean, default False
        Whether to return the number of iterations to convergence.

    penalize_latent: boolean, default True
        If false no penalisation is enforced on the latent sub-matrix.

    max_iter_graph_lasso: int, default 100
        Maximum number of iterations for the inner minimisation algorithm.

    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    log_likelihood: float,
        Likelihood of the final result.

    n_iter_ : int
        Number of iterations run.

    """
    h = M.shape[1]
    o = emp_cov.shape[0]
    emp_cov_H = np.zeros((h, h))
    emp_cov_OH = np.zeros_like(M)

    K = np.random.randn(h+o, h+o)
    K = K.dot(K.T)
    K /= np.max(K)

    if penalize_latent:
        regularizer = np.ones((h+o, h+o))
        regularizer -= np.diag(np.diag(regularizer))
        regularizer[:h, :h] *= eta
        regularizer[h:, h:] *= alpha
    else:
        regularizer = np.zeros((h+o, h+o))
        regularizer[h:, h:] = alpha*np.ones((o, o))
        regularizer[h:, h:] -= np.diag(np.diag(regularizer[h:, h:]))
    regularizer[:h, h:] = mu*M.T
    regularizer[h:, :h] = mu*M
    penalized_nll = np.inf

    checks = []
    for iter_ in range(max_iter):

        # expectation step
        sigma = pinvh(K)
        sigma_o_inv = pinvh(sigma[h:, h:])
        sigma_ho = sigma[:h, h:]
        sigma_oh = sigma[h:, :h]
        emp_cov_H = (sigma[:h, :h] - sigma_ho.dot(sigma_o_inv).dot(sigma_oh) +
                     np.linalg.multi_dot((sigma_ho, sigma_o_inv, emp_cov,
                                          sigma_o_inv, sigma_oh)))
        emp_cov_OH = emp_cov.dot(sigma_o_inv).dot(sigma_oh)
        S = np.zeros_like(K)
        S[:h, :h] = emp_cov_H
        S[:h, h:] = emp_cov_OH.T
        S[h:, :h] = emp_cov_OH
        S[h:, h:] = emp_cov
        penalized_nll_old = penalized_nll
        penalized_nll = _penalized_nll(K, S, regularizer)

        # maximization step
        K, _ = graphical_lasso(
            S, alpha=regularizer, rho=rho, return_n_iter=False,
            max_iter=max_iter_graph_lasso, verbose=int(max(verbose-1, 0)))

        check = convergence(obj=penalized_nll, rnorm=np.linalg.norm(K),
                            snorm=penalized_nll_old - penalized_nll,
                            e_pri=None, e_dual=None)

        checks.append((check, K, S))
        if verbose:
            print("iter: %d, NLL: %.6f , NLL_diff: %.6f" %
                  (iter_, check[0], check[2]))
        if np.abs(check[2]) < tol:
            break
    else:
        warnings.warn("The optimization of EM did not converged.")

    best_i = -1
    best_nll = np.inf
    for i, c in enumerate(checks):
        if c[0][0] < best_nll:
            best_nll = c[0][0]
            best_i = i
    K = checks[best_i][1]
    S = checks[best_i][2]
    penalized_nll_old = checks[0][0]
    return K, S, penalized_nll_old


class LatentMissingGraphicalLasso(GraphicalLasso):
    """Graphical Lasso with missing data as latent variables.

    This method allows for graphical model selection in presence of missing
    data in the dataset. It is suitable to estimate latent variables samples.
    For references see:
    "Yuan, Ming. Discussion: Latent variable graphical model selection via
    convex optimization. Ann. Statist. 40 (2012), no. 4, 1968--1972."

    "Tozzo, Veronica, et al. "Group induced graphical lasso allows for
    discovery of molecular pathways-pathways interactions." arXiv preprint
    arXiv:1811.09673 (2018)."

    Parameters
    ----------
    mask: array-like, shape=(n_dim_obs, n_dim_lat)
        Prior knowledge to put on the connections between latent and observed
        variables. If mask is a matrix of zeros the algorithm corresponds to
        the one of Yuan et al.(2012), in the other case to Tozzo et al.(2018).

    alpha : positive float, default 0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

    mu :  positive float, default 0.01
        The regularization parameter on the inter-links: the higher mu, the
        more the final matrix will have links similar to the ones in mask.

    eta :  positive float, default 0.1
        The regularization parameter on the latent variables: the higher eta,
        the sparser the network on the latent variables.

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

    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    """
    def __init__(self, mask, alpha=0.1, mu=0.1, eta=0.1, rho=1.,
                 tol=1e-4, rtol=1e-4, max_iter=100, verbose=False,
                 assume_centered=False,  update_rho=False,
                 random_state=None, compute_objective=True,
                 return_n_iter=True):
        GraphicalLasso.__init__(self, rho=rho,  max_iter=max_iter,
                                tol=tol, rtol=rtol, verbose=verbose,
                                assume_centered=assume_centered,
                                update_rho_options=update_rho,
                                compute_objective=compute_objective)
        self.mask = mask
        self.alpha = alpha
        self.mu = mu
        self.eta = eta
        self.random_state = random_state
        self.return_n_iter = return_n_iter

    def fit(self, X, y=None):
        """Fits the GraphLasso covariance model to X.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : (ignored)
        """

        self.random_state = check_random_state(self.random_state)
        X = check_array(X, ensure_min_features=2,
                        ensure_min_samples=2, estimator=self)

        if self.assume_centered:
            self.location_ = np.zeros((X.shape[0],  X.shape[1]))
        else:
            self.location_ = X.mean(0)

        emp_cov = empirical_covariance(
                        X, assume_centered=self.assume_centered)

        self.precision_, self.covariance_,  self.n_iter_ = \
            latent_missing_graphical_lasso(
                emp_cov, self.mask, lambda_=self.alpha, mu_=self.mu,
                eta_=self.eta, tol=self.tol,  max_iter=self.max_iter,
                verbose=self.verbose, compute_objective=self.compute_objective,
                return_n_iter=self.return_n_iter)
        return self


class MissingGraphicalLasso(GraphicalLasso):
    """Graphical Lasso with missing data.

    This method allows for graphical model selection in presence of missing
    data in the dataset. It is suitable to perform imputing after fitting.
    For references see "Mising values:sparse inverse covariance estimation and
    an extension to sparse regression", Stadler and Buhlman 2012
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
            self, alpha=0.01, rho=1., over_relax=1., max_iter=100,
            tol=1e-4, rtol=1e-4, verbose=False, assume_centered=False,
            update_rho_options=None,
            compute_objective=True, init='empirical',):
        super(MissingGraphicalLasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter, verbose=verbose,
            assume_centered=assume_centered, mode='admm', rho=rho,
            rtol=rtol, over_relax=over_relax,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective, init=init)

    def fit(self, X, y=None):
        """Fit the GraphicalLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : (ignored)

        """
        X, y = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, order="C",
            ensure_min_features=2, estimator=self,
            force_all_finite='allow-nan')

        if y is not None:
            # TODO
            warnings.warn('Not implemented')

        self.precision_, self.covariance_, self.complete_data_matrix_, \
            self.n_iter_ = missing_graphical_lasso(
                X, alpha=self.alpha, tol=self.tol, rtol=self.rtol,
                max_iter=self.max_iter, over_relax=self.over_relax,
                rho=self.rho, verbose=self.verbose, return_n_iter=True,
                return_history=False,
                update_rho_options=self.update_rho_options,
                compute_objective=self.compute_objective, init=self.init)
        return self
