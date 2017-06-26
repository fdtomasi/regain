import numpy as np
from numpy import zeros
from sklearn.covariance import empirical_covariance

from sklearn.utils.extmath import fast_logdet


def covsel(D, lamda, rho, alpha):
    # % covsel  Sparse inverse covariance selection via ADMM
    # % [X, history] = covsel(D, lambda, rho, alpha)
    # %
    # % Solves the following problem via ADMM:
    # %
    # %   minimize  trace(S*X) - log det X + lambda*||X||_1
    # %
    # % with variable X, where S is the empirical covariance of the data
    # % matrix D (training observations by features).
    # %
    # % The solution is returned in the matrix X.
    # %
    # % history is a structure that contains the objective value, the primal and
    # % dual residual norms, and the tolerances for the primal and dual residual
    # % norms at each iteration.
    # %
    # % rho is the augmented Lagrangian parameter.
    # %
    # % alpha is the over-relaxation parameter (typical values for alpha are
    # % between 1.0 and 1.8).
    # %
    # % More information can be found in the paper linked at:
    # % http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html

    QUIET    = 0;
    MAX_ITER = 1000;
    tol = 1e-4
    RELTOL   = 1e-2;

    S = empirical_covariance(D)
    n = S.shape[0]

    X = zeros(n)
    Z = zeros(n)
    U = zeros(n)

    # if ~QUIET
    #     fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
    #       'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
    # end
    hist = []
    for k in range(MAX_ITER):
        # % x-update
        es, Q = np.linalg.eigh(rho * (Z - U) - S)
        xi = (es + np.sqrt(es ** 2 + 4 * rho)) / (2. * rho)
        X = Q * np.diag(xi) * Q.T

        # % z-update with relaxation
        Zold = Z
        X_hat = alpha * X + (1 - alpha) * Zold
        Z = shrinkage(X_hat + U, lamda / rho)

        U = U + (X_hat - Z)

        # % diagnostics, reporting, termination checks
        history = []
        history.append(objective(S, X, Z, lamda))

        history.append(np.linalg.norm(X - Z, 'fro'))
        history.append(np.linalg.norm(-rho*(Z - Zold),'fro'))

        history.append(np.sqrt(n*n)*tol + RELTOL*max(np.linalg.norm(X,'fro'), np.linalg.norm(Z,'fro')))
        history.append(np.sqrt(n*n)*tol + RELTOL*np.linalg.norm(rho*U,'fro'))


        # if ~QUIET
        #     fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
        #         history.r_norm(k), history.eps_pri(k), ...
        #         history.s_norm(k), history.eps_dual(k), history.objval(k));
        # end

        hist.append(history)
        if history[1] < history[3] and history[2] < history[4]:
            break

    return Z


def objective(S, X, Z, lamda):
    return np.trace(S * X) - fast_logdet(X) + lamda * np.linalg.norm(Z, 1)


def shrinkage(a, kappa):
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)
