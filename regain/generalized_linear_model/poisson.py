import warnings

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.extmath import squared_norm
from sklearn.base import BaseEstimator

from regain.generalized_linear_model.base import GLM_GM, convergence
from regain.generalized_linear_model.base import build_adjacency_matrix
from regain.prox import soft_thresholding, soft_thresholding_od
from regain.norm import l1_od_norm
from regain.utils import convergence as convergence_admm


def objective_single_variable(X, theta, n, r, selector, alpha):
    objective = 0
    for i in range(X.shape[0]):
        XXT = X[i, r] * X[i, selector].dot(theta)
        XT = np.log(1 + np.exp(X[i, selector].dot(theta)))
        objective += XXT - XT
    return - (1/n) * objective + alpha*np.linalg.norm(theta, 1)


def fit_each_variable(X, ix, alpha=1e-2, gamma=1e-3, tol=1e-3,
                      max_iter=1000, verbose=0,
                      return_history=True, compute_objective=True,
                      return_n_iter=False, adjust_gamma=False):
    n, d = X.shape
    theta = np.zeros(d-1)
    selector = [i for i in range(d) if i != ix]

    def gradient(X, theta, r, selector, n):
        sum_ = 0
        for i in range(X.shape[0]):
            XT = X[i, selector].dot(theta)
            EXT = np.exp(XT)
            E_XT = np.exp(-XT)
            sum_ += X[i, selector]*((EXT - E_XT)/(EXT + E_XT) - X[i, r])
        return (1/n)*sum_

    thetas = [theta]
    checks = []
    for iter_ in range(max_iter):
        theta_new = theta - gamma*gradient(X, theta, ix, selector, n)
        theta = soft_thresholding(theta_new, alpha*gamma)
        thetas.append(theta)

        if iter_ > 0:
            check = convergence(iter=iter_,
                                obj=objective_single_variable(X, theta, n, ix,
                                                              selector, alpha),
                                iter_norm=np.linalg.norm(thetas[-2]-thetas[-1]),
                                iter_r_norm=(np.linalg.norm(thetas[-2] -
                                                            thetas[-1]) /
                                             np.linalg.norm(thetas[-1])))
            checks.append(check)
            # if adjust_gamma: # TODO multiply or divide
            if verbose:
                print('Iter: %d, objective: %.4f, iter_norm %.4f' %
                      (check[0], check[1], check[2]))

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
    if not np.all(theta == theta.T):
        return np.float('inf')
    for r in range(d):
        selector = [i for i in range(d) if i != r]
        for i in range(n):
            XXT = X[i, r] * X[i, selector].dot(theta[selector, r])
            XT = np.log(1 + np.exp(X[i, selector].dot(theta[selector, r])))
            objective += XXT - XT
    return objective


def objective(X, theta, alpha):
    n, _ = X.shape
    objective = loss(X, theta)
    return - (1/n) * objective + alpha*l1_od_norm(theta)


def _gradient_ising(X, theta,  n, A=None, rho=1, T=0):
    n, d = X.shape
    theta_new = np.zeros_like(theta)
    def gradient(X, thetas, r, selector, n, A=None, rho=1, T=0):
        sum_ = np.zeros((1, len(selector)))
        for i in range(X.shape[0]):
            XT = X[i, selector].dot(theta[selector, r])
            EXT = np.exp(XT)
            E_XT = np.exp(-XT)
            sum_ += X[i, selector]*((EXT - E_XT)/(EXT + E_XT) - X[i, r])
        if A is not None:
            sum_ += (rho*T/n)*(theta[r, selector] - A[r, selector])
        return (1/n)*sum_
    for ix in range(theta.shape[0]):
        selector = [i for i in range(d) if i != ix]
        theta_new[ix, selector] = gradient(
                                    X, theta, ix, selector, n, A, rho, T)
    theta_new = (theta_new + theta_new.T)/2
    return theta_new


def _fit(X, alpha=1e-2, gamma=1e-3, tol=1e-3, max_iter=1000, verbose=0,
         return_history=True, compute_objective=True, warm_start=None,
         return_n_iter=False, adjust_gamma=False, A=None, T=0, rho=1):
    n, d = X.shape
    if warm_start is None:
        theta = np.zeros((d, d))
    else:
        theta = check_array(warm_start)

    thetas = [theta]
    theta_new = theta.copy()
    checks = []
    for iter_ in range(max_iter):

        theta_new = theta - gamma*_gradient_ising(X, theta,  n, A, rho, T)
        theta = (theta_new + theta_new.T)/2
        theta = soft_thresholding_od(theta, alpha*gamma)
        thetas.append(theta)

        #assert np.all(np.diag(theta) == 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check = convergence(iter=iter_,
                                obj=objective(X, theta, alpha),
                                iter_norm=np.linalg.norm(thetas[-2]-thetas[-1]),
                                iter_r_norm=(np.linalg.norm(thetas[-2] -
                                                            thetas[-1]) /
                                             np.linalg.norm(thetas[-1])))
        checks.append(check)
        # if adjust_gamma: # TODO multiply or divide
        if verbose:
            print('Iter: %d, objective: %.4f, iter_norm %.4f' %
                  (check[0], check[1], check[2]))

        if np.abs(check[2]) < tol:
            break

    return_list = [thetas[-1]]
    if return_history:
        return_list.append(thetas)
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iter_)

    return return_list


def _fit_ADMM(X, alpha=1e-2, gamma=1e-3, tol=1e-3, rtol=1e-4, max_iter=1000,
              rho=1, verbose=0, return_history=True, compute_objective=True,
              return_n_iter=False, adjust_gamma=False):
    n, d = X.shape
    theta = np.zeros((d, d))

    def gradient(X, theta, A, U, r, selector, n, rho):
        sum_ = 0
        for i in range(X.shape[0]):
            XT = X[i, selector].dot(theta[r, selector].T)
            EXT = np.exp(XT)
            E_XT = np.exp(-XT)
            sum_ += X[i, selector]*((EXT - E_XT)/(EXT + E_XT) - X[i, r])

        sum_ += rho*(theta[r, selector] - D)
        return (1/n)*sum_

    thetas = [theta]
    theta_new = theta.copy()
    checks = []
    U = np.zeros_like(theta)
    A = np.zeros_like(theta)
    for iter_ in range(max_iter):

        theta = np.zeros_like(theta)
        old_iter = theta.copy()
        D = A - U
        D = (D+D.T)/2
        for inner_iter_ in range(max_iter//2):
            theta = theta -\
                gamma*_gradient_ising(X, theta,  n, A=D, rho=rho, T=1)
            #theta = (theta + theta.T)/2
            print(np.linalg.norm(theta - old_iter))
            if np.linalg.norm(theta - old_iter) < tol:
                break
            old_iter = theta_new.copy()

        thetas.append(theta)
        A = (theta + theta.T)/2
        A = soft_thresholding_od(A + U, alpha*gamma/rho)

        U += A - theta
        assert np.all(np.diag(A) == 0)

        check = convergence_admm(obj=objective(X, theta, alpha),
                                 rnorm=squared_norm(theta - A),
                                 snorm=(squared_norm(thetas[-2] - thetas[-1])),
                                 e_pri=(np.sqrt(theta.size)*tol +
                                        rtol*max(squared_norm(theta),
                                                 squared_norm(A))),
                                 e_dual=(np.sqrt(theta.size)*tol +
                                         rtol*rho*np.linalg.norm(U)))
        checks.append(check)
        # if adjust_gamma: # TODO multiply or divide
        if verbose:
            print(
                "obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        if check[1] < check[3] and check[2] < check[4]:
            break

    return_list = [A]
    if return_history:
        return_list.append(thetas)
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iter_)

    return return_list


class Ising_GLM_GM(GLM_GM, BaseEstimator):

    def __init__(self, alpha=0.01, tol=1e-4, rtol=1e-4, reconstruction='union',
                 mode='coordinate_descent', rho=1, max_iter=100,
                 verbose=False, return_history=True, return_n_iter=False,
                 compute_objective=True):
        super(Ising_GLM_GM, self).__init__(
            alpha, tol, rtol, max_iter, verbose, return_history, return_n_iter,
            compute_objective)
        self.reconstruction = reconstruction
        self.mode = mode
        self.rho = rho

    def get_precision(self):
        return self.precision_

    def fit(self, X, y=None, gamma=0.1):
        """
        X : ndarray, shape = (n_samples * n_times, n_dimensions)
            Data matrix.
        y : added for compatiblity
        gamma: float,
            Step size of the proximal gradient descent.
        """
        X = check_array(X)
        if self.mode.lower() == 'symmetric_fbs':
            res = _fit(X, self.alpha, tol=self.tol, gamma=gamma,
                       max_iter=self.max_iter,
                       verbose=self.verbose)
            self.precision_ = res[0]
            self.history = res[1:]
        elif self.mode.lower() == 'coordinate_descent':
            thetas_pred = []
            historys = []
            for ix in range(X.shape[1]):
                # TODO: livello di verbosita'
                res = fit_each_variable(X, ix, self.alpha, tol=self.tol,
                                        gamma=gamma,
                                        verbose=self.verbose)
                thetas_pred.append(res[0])
                historys.append(res[1:])
            self.precision_ = build_adjacency_matrix(thetas_pred,
                                                     how=self.reconstruction)
            self.history = historys
        elif self.mode.lower() == 'admm':
            res = _fit_ADMM(X, self.alpha, rho=self.rho, tol=self.tol,
                            rtol=self.rtol, gamma=gamma,
                            max_iter=self.max_iter,
                            verbose=self.verbose)
            self.precision_ = res[0]
            self.history = res[1:]
        else:
            raise ValueError('Unknown optimization mode. Found ' + self.mode +
                             ". Options are 'coordiante_descent', "
                             "'symmetric_fbs'")
        return self

    def score(self, X, y=None):
        return 0
