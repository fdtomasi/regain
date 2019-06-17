import warnings
import numpy as np

from scipy.linalg import pinvh
from functools import partial

from sklearn.covariance import empirical_covariance
from sklearn.utils import check_X_y
from regain.covariance.time_graphical_lasso_ import TimeGraphicalLasso
from regain.covariance.time_graphical_lasso_ import time_graphical_lasso, loss
from regain.multi_layer.two_layers_graphical_lasso import \
                                        TwoLayersGraphicalLasso
from regain.scores import log_likelihood_t, BIC_t, EBIC_t, EBIC_m_t
from regain.validation import check_norm_prox
from regain.utils import convergence, ensure_posdef, positive_definite
from regain.norm import l1_norm
# X = np.vstack(data_list)
#     y = np.array([np.ones(x.shape[0]) * i
#                   for i, x in enumerate(data_list)]).flatten().astype(int)


def objective(K, S, n_samples, alpha, beta, psi):
    obj = loss(S, K, n_samples=n_samples)
    obj += sum(map(l1_norm, alpha * K))
    obj += beta * sum(map(psi, K[1:] - K[:-1]))
    return obj


def two_layers_time_graphical_lasso(
        emp_cov, h=2, alpha=0.01, M=None, mu=0, eta=0, beta=1.,
        psi="laplacian",
        n_samples=None, assume_centered=False, tol=1e-3, rtol=1e-3,
        max_iter=200, verbose=0, rho=1., compute_objective=False,
        return_history=False, return_n_iter=False):

    psi_func, _, _ = check_norm_prox("laplacian")
    if M is None:
        M = np.zeros((emp_cov[0].shape[0], h))
    else:
        h = M.shape[1]
    o = emp_cov[0].shape[0]

    Ks = [np.random.randn(h+o, h+o) for i in range(emp_cov.shape[0])]
    Ks = [K.dot(K.T) for K in Ks]
    Ks = [K / np.max(K) for K in Ks]
    Ks = np.array(Ks)

    regularizer = np.ones((h+o, h+o))
    regularizer -= np.diag(np.diag(regularizer))
    regularizer[:h, :h] *= eta
    regularizer[h:, h:] *= alpha
    regularizer[:h, h:] = mu*M.T
    regularizer[h:, :h] = mu*M
    penalized_nll = np.inf

    checks = []
    Ss = np.zeros((emp_cov.shape[0], h+o, h+o))
    Ks_prev = None
    for iter_ in range(max_iter):

        # expectation step
        Ks_prev = Ks.copy()
        Ss_ = []
        for i, K in enumerate(Ks):
            try:
                sigma = np.linalg.inv(K)
                sigma_o_inv = np.linalg.inv(sigma[h:, h:])
            except:
                sigma = pinvh(K)
                sigma_o_inv = pinvh(sigma[h:, h:])
            sigma_ho = sigma[:h, h:]
            sigma_oh = sigma[h:, :h]
            emp_cov_H = (sigma[:h, :h] - sigma_ho.dot(sigma_o_inv).dot(
                        sigma_oh) + np.linalg.multi_dot((sigma_ho, sigma_o_inv,
                        emp_cov[i], sigma_o_inv, sigma_oh)))
            emp_cov_OH = emp_cov[i].dot(sigma_o_inv).dot(sigma_oh)
            S = np.zeros_like(K)
            S[:h, :h] = emp_cov_H
            S[:h, h:] = emp_cov_OH.T
            S[h:, :h] = emp_cov_OH
            S[h:, h:] = emp_cov[i]
            Ss_.append(S)

        Ss = np.array(Ss_)

        # maximization step
        res = time_graphical_lasso(
                Ss, alpha=regularizer, rho=rho, beta=beta, max_iter=max_iter,
                n_samples=n_samples,
                verbose=int(max(verbose-1, 0)), psi=psi, tol=1e-3, rtol=rtol,
                return_history=return_history,  return_n_iter=return_n_iter,
                init=Ks)
        Ks = res[0]
        penalized_nll_old = penalized_nll
        penalized_nll = objective(Ks, Ss, n_samples, regularizer, beta,
                                  psi_func)

        check = convergence(obj=penalized_nll, rnorm=np.linalg.norm(K),
                            snorm=penalized_nll_old - penalized_nll,
                            e_pri=None, e_dual=None)

        checks.append(check)
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
    return_values = [Ks, Ss, h, penalized_nll_old]
    if return_history:
        return_values.append(checks)
    if return_n_iter:
        return_values.append(iter_)
    return return_values


class TwoLayersTimeGraphicalLasso(TwoLayersGraphicalLasso, TimeGraphicalLasso):
    """
    TODO docstrings
    """
    def __init__(self, h=2, mask=None, alpha=0.1, mu=0, eta=0, beta=1.,
                 rho=1., psi='laplacian', n_samples=None, tol=1e-4, rtol=1e-4,
                 max_iter=100, verbose=0,
                 update_rho=False, random_state=None,
                 score_type='likelihood',
                 assume_centered=False,
                 compute_objective=True):
        TwoLayersGraphicalLasso.__init__(self, mask, mu=mu,
                                         eta=eta, random_state=random_state)
        TimeGraphicalLasso.__init__(self, alpha=alpha, beta=beta, rho=rho,
                                    tol=tol, rtol=rtol, psi=psi,
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
        # TODO check su H
        # n_samples = np.array([x.shape[0] for x in X])
        if self.assume_centered:
            self.location_ = np.zeros((n_times, n_dimensions))
        else:
            self.location_ = np.array(
                [X[y == cl].mean(0) for cl in self.classes_])

        emp_cov = np.array([empirical_covariance(X[y == cl],
                            assume_centered=self.assume_centered)
                            for cl in self.classes_])

        self.precision_, self.covariance_, self.n_latent_, self.objective_, \
            self.history_, self.iters_ = two_layers_time_graphical_lasso(
                emp_cov, h=self.h, alpha=self.alpha, M=self.mask, mu=self.mu,
                eta=self.eta, beta=self.beta, psi=self.psi,
                # n_samples=self.n_samples,
                assume_centered=self.assume_centered,
                tol=self.tol, rtol=self.rtol,
                max_iter=self.max_iter, verbose=self.verbose, rho=self.rho,
                compute_objective=self.compute_objective,
                return_history=True,
                return_n_iter=True)
        return self

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
        precision = np.array([p[self.n_latent_:, self.n_latent_:]
                              for p in self.precision_])
        if not positive_definite(precision):
            ensure_posdef(precision)
        precision = [p for p in precision]
        s = score_func(emp_cov, precision)
        return s
