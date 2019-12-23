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
from functools import partial

from sklearn.covariance import empirical_covariance
from sklearn.utils.validation import check_X_y

from regain.covariance.missing_graphical_lasso_ import \
        _compute_empirical_covariance, _compute_cs, _compute_mean
from regain.covariance.kernel_time_graphical_lasso_ import \
                kernel_time_graphical_lasso, KernelTimeGraphicalLasso
from regain.covariance.time_graphical_lasso_ import loss
from regain.covariance.missing_graphical_lasso import \
                                        LatentMissingGraphicalLasso
from regain.scores import log_likelihood_t, BIC_t, EBIC_t, EBIC_m_t
from regain.validation import check_norm_prox
from regain.utils import convergence, ensure_posdef, positive_definite
from regain.norm import l1_norm


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


def objective(K, S, n_samples, alpha, beta, psi):
    obj = loss(S, K, n_samples=n_samples)
    obj += sum(map(l1_norm, alpha * K))
    obj += beta * sum(map(psi, K[1:] - K[:-1]))
    return obj


def latent_missing_time_graphical_lasso(
        emp_cov, h=2, alpha=0.01, M=None, mu=0, eta=0, beta=1., kernel=None,
        psi="laplacian", strong_M=False,
        n_samples=None, assume_centered=False, tol=1e-3, rtol=1e-3,
        max_iter=200, verbose=0, rho=1., compute_objective=False,
        return_history=False, return_n_iter=False):

    psi_func, _, _ = check_norm_prox(psi)
    if M is None:
        M = np.zeros((emp_cov[0].shape[0], h))
    else:
        h = M.shape[1]
    o = emp_cov[0].shape[0]

    Ks = [np.random.randn(h+o, h+o)*1.5 for i in range(emp_cov.shape[0])]
    Ks = [K.dot(K.T) for K in Ks]
    Ks = [K / np.max(K) for K in Ks]
    Ks = np.array(Ks)

    if strong_M:
        for i in range(Ks.shape[0]):
            Ks[i, :h, h:] = M.T
            Ks[i, h:, :h] = M

    regularizer = np.ones((h+o, h+o))
    regularizer -= np.diag(np.diag(regularizer))
    regularizer[:h, :h] *= eta
    regularizer[h:, h:] *= alpha
    if strong_M:
        regularizer[:h, h:] = 0
        regularizer[h:, :h] = 0
    else:
        regularizer[:h, h:] = mu*M.T
        regularizer[h:, :h] = mu*M
    penalized_nll = np.inf

    checks = []
    Ss = np.zeros((emp_cov.shape[0], h+o, h+o))
    Ks_prev = None
    likelihoods = []
    for iter_ in range(max_iter):

        # expectation step
        Ks_prev = Ks.copy()
        Ss_ = []
        for i, K in enumerate(Ks):
            if strong_M:
                K[:h, h:] = M.T
                K[h:, :h] = M

            S = np.zeros_like(K)
            K_inv = np.linalg.pinv(K[:h, :h])
            S[:h, :h] = K_inv + K_inv.dot(K[:h, h:]).dot(
                            emp_cov[i]).dot(K[h:, :h]).dot(K_inv)
            S[:h, h:] = K_inv.dot(K[:h, h:].dot(emp_cov[i]))
            S[h:, :h] = S[:h, h:].T
            S[h:, h:] = emp_cov[i]

            Ss_.append(S)

        Ss = np.array(Ss_)
        Ks = kernel_time_graphical_lasso(
                Ss, alpha=alpha, rho=rho, kernel=kernel,
                max_iter=max_iter, verbose=max(0, verbose-1),
                psi=psi, tol=tol, rtol=tol,
                return_history=False, return_n_iter=True, mode='admm',
                update_rho_options=None, compute_objective=False, stop_at=None,
                stop_when=1e-4, init='empirical')[0]

        penalized_nll_old = penalized_nll
        penalized_nll = objective(Ks, Ss, n_samples, regularizer, beta,
                                  psi_func)

        check = convergence(obj=penalized_nll, rnorm=np.linalg.norm(K),
                            snorm=penalized_nll_old - penalized_nll,
                            e_pri=None, e_dual=None)

        checks.append(check)
        thetas = [k[h:, h:] -
                  k[h:, :h].dot(np.linalg.pinv(k[:h, :h])).dot(k[:h, h:])
                  for k in Ks]
        likelihoods.append(log_likelihood_t(emp_cov, thetas))
        if verbose:
            print("iter: %d, NLL: %.6f , NLL_diff: %.6f" %
                  (iter_, check[0], check[2]))
        if iter_ > 2:
            if np.abs(check[2]) < tol:
                break
            if check[2] < 0 and checks[-2][2] > 0:
                Ks = Ks_prev
                break
    else:
        warnings.warn("The optimization of EM did not converged.")
    returns = [Ks, likelihoods]
    if return_n_iter:
        returns.append(iter_)
    return returns


class TwoLayersTimeGraphicalLasso(LatentMissingGraphicalLasso,
                                  KernelTimeGraphicalLasso):
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

    kernel : ndarray, default None
        Normalised temporal kernel (1 on the diagonal),
        with dimensions equal to the dimensionality of the data set.
        If None, it is interpreted as an identity matrix, where there is no
        constraint on the temporal behaviour of the precision matrices.

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
    def __init__(self, h=2, mask=None, alpha=0.1, mu=0, eta=0, beta=1.,
                 rho=1., psi='laplacian', n_samples=None, tol=1e-4, rtol=1e-4,
                 max_iter=100, verbose=0, kernel=None,
                 update_rho=False, random_state=None,
                 score_type='likelihood',
                 assume_centered=False,
                 compute_objective=True):
        LatentMissingGraphicalLasso.__init__(self, mask, mu=mu,
                                             eta=eta,
                                             random_state=random_state)
        KernelTimeGraphicalLasso.__init__(
                                    self, alpha=alpha, beta=beta, rho=rho,
                                    tol=tol, rtol=rtol, psi=psi, kernel=kernel,
                                    max_iter=max_iter,
                                    assume_centered=assume_centered)
        self.score_type = score_type
        self.h = h
        self.verbose = verbose
        self.compute_objective = compute_objective

    def fit(self, X, y):
        """Fit the KernelTimeGraphLasso model to X.

        Parameters
        ----------
        X : ndarray, shape = (n_samples * n_times, n_dimensions)
            Data matrix.
        y : ndarray, shape = (n_times,)
            Indicate the temporal belonging of each sample.
        """

        # Covariance does not make sense for a single feature
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64,
                         order="C", ensure_min_features=2, estimator=self)

        n_dimensions = X.shape[1]
        self.classes_, n_samples = np.unique(y, return_counts=True)
        n_times = self.classes_.size

        if self.assume_centered:
            self.location_ = np.zeros((n_times, n_dimensions))
        else:
            self.location_ = np.array(
                [X[y == cl].mean(0) for cl in self.classes_])

        emp_cov = np.array([empirical_covariance(X[y == cl],
                            assume_centered=self.assume_centered)
                            for cl in self.classes_])

        self.precision_, _, self.n_iter_ = latent_missing_time_graphical_lasso(
                emp_cov, h=self.h, alpha=self.alpha, M=self.mask, mu=self.mu,
                eta=self.eta, beta=self.beta, psi=self.psi, kernel=self.kernel,
                assume_centered=self.assume_centered,
                tol=self.tol, rtol=self.rtol,
                max_iter=self.max_iter, verbose=self.verbose, rho=self.rho,
                compute_objective=self.compute_objective,
                return_history=True,
                return_n_iter=True)
        return self

    def get_observed_precision(self):
        precision = []
        for p in self.precision_:
            obs = p[self.n_latent_:, self.n_latent_:]
            lat = p[:self.n_latent_, :self.n_latent_]
            inter = p[:self.n_latent_, self.n_latent_:]
            precision.append(obs - inter.T.dot(np.linalg.pinv(lat)).dot(inter))
        return np.array(precision)

    def score(self, X, y):
        n = X.shape[0]
        emp_cov = [empirical_covariance(X[y == cl] - self.location_[i],
                   assume_centered=True) for i, cl in enumerate(self.classes_)]
        score_func = {'likelihood': log_likelihood_t,
                      'bic': BIC_t,
                      'ebic': partial(EBIC_t, n=n),
                      'ebicm': partial(EBIC_m_t, n=n)}
        try:
            score_func = score_func[self.score_type]
        except KeyError:
            warnings.warn("The score type passed is not available, using log "
                          "likelihood.")
            score_func = log_likelihood_t
        precision = self.get_observed_precision()
        if not positive_definite(precision):
            ensure_posdef(precision)
        precision = [p for p in precision]
        s = score_func(emp_cov, precision)
        return s


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
                rtol=self.rtol, kernel=self.kernel,
                psi=self.psi, return_n_iter=True,
                update_rho_options=self.update_rho_options,
                compute_objective=self.compute_objective)
        return self
