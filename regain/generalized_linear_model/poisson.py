import numpy as np


from sklearn.utils import check_array


from regain.generalized_linear_model.base import GLM_GM, convergence, \
                                                 build_adjacency_matrix
from regain.prox import soft_thresholding


def objective(X, theta, n, r, selector, alpha):
    XXT = X[:, r].T.dot(X[:, selector]).dot(theta)
    TXXT = theta.T.dot(X[:, selector].T).dot(X[:, selector]).dot(theta)
    return -(1/n)*XXT + (1/(2*n)) * TXXT + alpha * np.linalg.norm(theta, 1)


def fit_each_variable(X, ix, alpha=1e-2, gamma=1e-3, tol=1e-3,
                      max_iter=1000, verbose=0,
                      return_history=True, compute_objective=True,
                      return_n_iter=False, adjust_gamma=False):
    n, d = X.shape
    theta = np.zeros(d-1)
    selector = [i for i in range(d) if i != ix]

    def gradient(X, theta, r, selector, n):
        XX = X[:, r].T.dot(X[:, selector])
        XXT = X[:, selector].T.dot(X[:, selector]).dot(theta)
        return - (1/n) * XX + (1/n) * XXT

    thetas = [theta]
    checks = []
    for iter_ in range(max_iter):
        theta_new = theta - gamma*gradient(X, theta, ix, selector, n)
        theta = soft_thresholding(theta_new, alpha*gamma)
        thetas.append(theta)

        check = convergence(iter=iter_,
                            obj=objective(X, theta, n, ix, selector, alpha),
                            iter_norm=np.linalg.norm(thetas[-2]-thetas[-1]),
                            iter_r_norm=(np.linalg.norm(thetas[-2] -
                                                        thetas[-1]) /
                                         np.linalg.norm(thetas[-1])))
        checks.append(check)
        # if adjust_gamma: # TODO multiply or divide
        if verbose:
            print('Iter: %d, objective: %.4f, iter_norm %.4f' %
                  (check[0], check[1], check[2]))

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

    def __init__(self, alpha=0.01, tol=1e-4, rtol=1e-4, max_iter=100,
                 verbose=False, return_history=True, return_n_iter=False,
                 compute_objective=True):
        super(Gaussian_GLM_GM, self).__init__(
            alpha, tol, rtol, max_iter, verbose, return_history, return_n_iter,
            compute_objective)

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
        self.precision_ = build_adjacency_matrix(thetas_pred)
        self.history = historys
        return self
