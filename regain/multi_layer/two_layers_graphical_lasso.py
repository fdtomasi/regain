import warnings
import operator
import numpy as np

from functools import partial
from itertools import product
from scipy.linalg import pinvh

from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import fast_logdet
from sklearn.covariance import empirical_covariance
from sklearn.model_selection import check_cv
from sklearn.utils import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from regain.covariance import  GraphicalLasso
from regain.covariance.graphical_lasso_ import graphical_lasso
from regain.scores import log_likelihood, BIC, EBIC, EBIC_m
from regain.utils import convergence


def get_params_list(params, emp_cov, random_search, random_state):
    if isinstance(params, list):
        return params
    elif isinstance(params, tuple):
        if random_search:
            params = np.array([random_state.uniform(params[0],
                               params[1]) for i in range(params)])
        else:
            params = np.array([np.logspace(np.log10(params[0]),
                                           np.log10(params[1]))
                              for i in range(params)])
    else:
        par_1 = par_max(emp_cov)
        par_0 = 1e-2 * par_1
        if random_search:
            params = np.array([random_state.uniform(par_0,
                               par_1) for i in range(params)])
        else:
            params = np.logspace(np.log10(par_0), np.log10(par_1),
                                 params)[::-1]
    return params


def par_max(emp_cov):
    A = np.copy(emp_cov)
    A.flat[::A.shape[0] + 1] = 0
    return np.max(np.abs(A))


def _penalized_nll(K, S=None, regularizer=None):
    res = - fast_logdet(K) + np.sum(K*S)
    res += np.linalg.norm(regularizer*K, 1)
    return res


def tlgl_path(X_train, mask=None, lambdas=[0.1], mus=[0.1], etas=[0.1],
              X_test=None, random_search=True,  tol=1e-3, max_iter=200,
              update_rho=False, verbose=0, score='ebic',
              random_state=None, save_all=False, penalize_latent=True):

    score_func = {'likelihood': log_likelihood,
                  'bic': BIC,
                  'ebic': partial(EBIC, n=X_test.shape[0]),
                  'ebicm': partial(EBIC_m, n=X_test.shape[0])}
    try:
        score_func = score_func[score]
    except KeyError:
        warnings.warn("The score type passed is not available, using log "
                      "likelihood.")
        score_func = log_likelihood

    emp_cov = empirical_covariance(X_train)

    covariances_ = list()
    precisions_ = list()
    observeds_ = list()
    scores_ = list()

    if X_test is not None:
        test_emp_cov = empirical_covariance(X_test)

    if random_search:
        couples = zip(lambdas, mus, etas)
    else:
        couples = product(lambdas, mus, etas)
    for i, params in enumerate(couples):
        print(params)
        try:
            # Capture the errors, and move on
            prec_, cov_, _ = two_layers_graphical_lasso(
                emp_cov, mask, params[0], params[1], params[2],
                max_iter=max_iter, return_n_iter=False,
                penalize_latent=penalize_latent)
            if save_all:
                covariances_.append(cov_)
                precisions_.append(prec_)

            h = mask.shape[1]
            observed = prec_[h:, h:] - \
                prec_[h:, :h].dot(pinvh(prec_[:h, :h])).dot(prec_[:h, h:])
            observeds_.append(observed)
        except FloatingPointError:
            this_score = -np.inf
        if save_all:
            covariances_.append(np.nan)
            precisions_.append(np.nan)
            observeds_.append(np.nan)
        if X_test is not None:
            this_score = score_func(test_emp_cov, observed)
            if not np.isfinite(this_score):
                this_score = -np.inf
            scores_.append(this_score)
        if verbose:
            if X_test is not None:
                print("[graphical_lasso_path] lambda_: %.2f, mu: %.2f, "
                      "eta:%.2f, score: %.2f"
                      % (params[0], params[1], params[2], this_score))
            else:
                print('[graphical_lasso_path]  lambda_: %.2f, mu: %.2f, '
                      'eta:%.2f.' %
                      (params[0], params[1], params[2]))
    if X_test is not None:
        return covariances_, precisions_, observeds_, scores_
    return covariances_, precisions_, observeds_


def two_layers_graphical_lasso(emp_cov, M, lambda_, mu_, eta_=0, rho=1,
                               assume_centered=False, tol=1e-3, max_iter=200,
                               verbose=0, compute_objective=False,
                               return_n_iter=False, penalize_latent=True,
                               max_iter_graph_lasso=100):
    """
    # TODO docstrings
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
        regularizer[:h, :h] *= eta_
        regularizer[h:, h:] *= lambda_
    else:
        regularizer = np.zeros((h+o, h+o))
        regularizer[h:, h:] = lambda_*np.ones((o, o))
        regularizer[h:, h:] -= np.diag(np.diag(regularizer[h:, h:]))
    regularizer[:h, h:] = mu_*M.T
    regularizer[h:, :h] = mu_*M
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
        K, _ = graphical_lasso(S, alpha=regularizer, rho=rho, return_n_iter=False,
                           max_iter=max_iter_graph_lasso,
                           verbose=int(max(verbose-1, 0)))

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


class TwoLayersGraphicalLasso(GraphicalLasso):
    """
    todo: docstrings
    """
    def __init__(self, mask, lamda=0.1, mu=0.1, eta=0.1, rho=1.,
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
        self.lamda = lamda
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
            two_layers_graphical_lasso(
                emp_cov, self.mask, lambda_=self.lamda, mu_=self.mu,
                eta_=self.eta, tol=self.tol,  max_iter=self.max_iter,
                verbose=self.verbose, compute_objective=self.compute_objective,
                return_n_iter=self.return_n_iter)
        return self


class TwoLayersGraphicalLassoCV(TwoLayersGraphicalLasso):
    """
    mask: array-like, shape=()
        The regularization mask
    lambdas: list or integer, optional
        The parameters to test.
        If an integer n is passed then n parameters are selected in a
        (hopefully) suitable interval.
        If a list is passed then the parameters in the list are used to test
        the method.
        If a tuple of two elements is passed 10 elements between the elements
        of the tuple are tested.
    mus: the parameters to test
        If an integer n is passed then n parameters are selected in a
        (hopefully) suitable interval.
        If a list is passed then the parameters in the list are used to test
        the method.
        If a tuple of two elements is passed 10 elements between the elements
        of the tuple are tested.
    etas:  list or integer, optional
        The parameters to test.
        If an integer n is passed then n parameters are selected in a
        (hopefully) suitable interval.
        If a list is passed then the parameters in the list are used to test
        the method.
        If a tuple of two elements is passed 10 elements between the elements
        of the tuple are tested.
    score: string, optional
        Type of scores to use for testing.
    cv:
    tol: float, optional
    max_iter: int, optional
    n_jobs: int, optional
    random_search: boolean, default True
        If True and lambdas and mus are either an integer or a tuple, the
        values are randomly selected, not in a grid search.
    verbose: boolearn, ptional
    assume_centered: boolean, optional
    update_rho: boolean, optional
    random_state: boolean, ptional
    n_params: boolean, ptional
    save_all fdf
    penalized_latent
    """
    def __init__(self, mask, lambdas=4, mus=4, etas=4, rho=1., score='ebic',
                 cv=None, tol=1e-4, max_iter=100, n_jobs=1, random_search=True,
                 verbose=False, assume_centered=False,  update_rho=False,
                 random_state=None, n_params=10, save_all=False,
                 penalize_latent=True):
        self.mask = mask
        self.lambdas = lambdas
        self.mus = mus
        self.rho = rho
        self.etas = etas
        self.cv = cv
        self.n_jobs = n_jobs
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.update_rho = update_rho
        self.assume_centered = assume_centered
        self.random_state = random_state
        self.score = score
        self.random_search = random_search
        self.n_params = n_params
        self.save_all = save_all
        self.penalize_latent = penalize_latent

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

        self.X_train = X
        if self.assume_centered:
            self.location_ = np.zeros((X.shape[0],  X.shape[1]))
        else:
            self.location_ = X.mean(0)

        emp_cov = empirical_covariance(
                        X, assume_centered=self.assume_centered)

        cv = check_cv(self.cv, y, classifier=False)

        path = list()
        inner_verbose = max(0, self.verbose - 1)
        lambdas = get_params_list(self.lambdas, emp_cov, self.random_search,
                                  self.random_state)
        mus = get_params_list(self.mus, emp_cov, self.random_search,
                              self.random_state)
        etas = get_params_list(self.lambdas, emp_cov, self.random_search,
                               self.random_state)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            this_path = Parallel(n_jobs=self.n_jobs, verbose=inner_verbose)(
                        delayed(tlgl_path)(
                            X[train], mask=self.mask, lambdas=lambdas, mus=mus,
                            etas=etas, X_test=X[test], tol=self.tol,
                            max_iter=int(.1 * self.max_iter),
                            random_search=self.random_search,
                            update_rho=self.update_rho, verbose=0,
                            random_state=self.random_state, score=self.score,
                            save_all=self.save_all,
                            penalize_latent=self.penalize_latent)
                        for train, test in cv.split(X, y))

        # Little danse to transform the list in what we need
        covs, precs, hidds, scores = zip(*this_path)

        if self.save_all:
            covs = zip(*covs)
            precs = zip(*precs)
            hidds = zip(*hidds)
            scores = zip(*scores)

        combinations = list(zip(lambdas, mus, etas))
        if self.save_all:
            path.extend(zip(combinations, scores, covs))
        else:
            path.extend(zip(combinations, scores))
        path = sorted(path, key=operator.itemgetter(0), reverse=True)

        # Find the maximum (avoid using built in 'max' function to
        # have a fully-reproducible selection of the smallest alpha
        # in case of equality)
        best_score = -np.inf
        for index, (combination, scores) in enumerate(path):
            this_score = np.mean(scores)
            if this_score >= .1 / np.finfo(np.float64).eps:
                this_score = np.nan
            if this_score >= best_score:
                best_score = this_score
                best_index = index

        path = list(zip(*path))
        self.grid_scores = list(path[1])
        best_lambda, best_mu, best_eta = combinations[best_index]
        self.lambda_ = best_lambda
        self.mu_ = best_mu
        self.eta_ = best_eta
        self.cv_parameters_ = combinations

        # Finally fit the model with the selected alpha
        self.precision_, self.covariance_,  self.n_iter_ = \
            two_layers_graphical_lasso(emp_cov, self.mask, lambda_=best_lambda,
                                       mu_=best_mu, eta_=best_eta,
                                       tol=self.tol,  max_iter=self.max_iter,
                                       verbose=self.verbose,
                                       compute_objective=True,
                                       return_n_iter=True)
        return self

    def grid_scores(self):
        return self.grid_scores_
