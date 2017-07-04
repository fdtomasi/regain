from __future__ import division

import numpy as np
from numpy import zeros
from sklearn.covariance import empirical_covariance

from sklearn.utils.extmath import fast_logdet


def covsel(D, lamda=1, rho=1, alpha=1, max_iter=1000, verbose=False, tol=1e-4,
           rtol=1e-2):
    # covsel  Sparse inverse covariance selection via ADMM
    #  [X, history] = covsel(D, lambda, rho, alpha)
    #
    #  Solves the following problem via ADMM:
    #
    #    minimize  trace(S*X) - log det X + lambda*||X||_1
    #
    #  with variable X, where S is the empirical covariance of the data
    #  matrix D (training observations by features).
    #
    #  The solution is returned in the matrix X.
    #
    #  history is a structure that contains the objective value, the primal and
    #  dual residual norms, and the tolerances for the primal and dual residual
    #  norms at each iteration.
    #
    #  rho is the augmented Lagrangian parameter.
    #
    #  alpha is the over-relaxation parameter (typical values for alpha are
    #  between 1.0 and 1.8).
    #
    #  More information can be found in the paper linked at:
    #  http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    #
    S = empirical_covariance(D)
    n = S.shape[0]

    # X = zeros(n)
    Z = zeros((n, n))
    U = zeros((n, n))

    # if ~QUIET
    #     fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
    #       'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
    # end
    hist = []
    count = 0
    for k in range(max_iter):
        # x-update
        es, Q = np.linalg.eigh(rho * (Z - U) - S)
        xi = (es + np.sqrt(es ** 2 + 4 * rho)) / (2. * rho)
        X = np.dot(Q.dot(np.diag(xi)), Q.T)

        # z-update with relaxation
        Zold = Z
        X_hat = alpha * X + (1 - alpha) * Zold
        Z = shrinkage(X_hat + U, lamda / rho)

        U = U + (X_hat - Z)

        # diagnostics, reporting, termination checks
        history = (
            objective(S, X, Z, lamda),

            np.linalg.norm(X - Z, 'fro'),
            np.linalg.norm(-rho * (Z - Zold), 'fro'),

            np.sqrt(n) * tol + rtol * max(
                np.linalg.norm(X, 'fro'), np.linalg.norm(Z, 'fro')),
            np.sqrt(n) * tol + rtol * np.linalg.norm(rho * U, 'fro')
        )

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % history)

        hist.append(history)
        if history[1] < history[3] and history[2] < history[4]:
            if count > 10:
                break
            else:
                count += 1
        else:
            count = 0

    return X, Z, hist


def objective(S, X, Z, lamda):
    return np.trace(S.dot(X)) - fast_logdet(X) + lamda * np.linalg.norm(Z, 1)


def shrinkage(a, kappa):
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)
