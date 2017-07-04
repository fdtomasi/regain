import numpy as np
import time


def lasso(A, b, lamda=1.0, rho=1.0, alpha=1.0, max_iter=1000, tol=1e-4):
    # % lasso  Solve lasso problem via ADMM
    # %
    # % [z, history] = lasso(A, b, lambda, rho, alpha);
    # %
    # % Solves the following problem via ADMM:
    # %   minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1
    # %
    # % The solution is returned in the vector x.
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
    # %
    # % More information can be found in the paper linked at:
    # % http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    # %
    # t_start = time.time()

    RELTOL = 1e-2

    m, n = A.shape

    # % save a matrix-vector multiply
    Atb = A.T.dot(b)

    # ADMM solver
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    # % cache the factorization
    L, U = lu_factor(A, rho)

    hist = []
    for k in range(max_iter):
        # % x-update
        q = Atb + rho * (z - u)  # % temporary value
        if m >= n:
            x = np.linalg.lstsq(U, np.linalg.lstsq(L, q)[0])[0]
        else:
            x = q - A.T.dot(
                np.linalg.lstsq(
                    U, np.linalg.lstsq(
                        L, A.dot(q))[0])[0]) / rho
            x /= rho

        # % z-update with relaxation
        zold = z
        x_hat = alpha * x + (1 - alpha) * zold
        z = shrinkage(x_hat + u, lamda / rho)

        # % u-update
        u += (x_hat - z)

        # % diagnostics, reporting, termination checks
        history = (
            objective(A, b, lamda, x, z),  # obj

            np.linalg.norm(x - z),  # r norm
            np.linalg.norm(-rho * (z - zold)),  # s norm

            np.sqrt(n) * tol + RELTOL * max(
                np.linalg.norm(x), np.linalg.norm(-z)),  # eps pri
            np.sqrt(n) * tol + RELTOL * np.linalg.norm(rho * u)  # eps dual
        )

        hist.append(history)
        if history[1] < history[3] and history[2] < history[4]:
            break

    return z


def objective(A, b, alpha, x, z):
    return .5 * np.sum((A.dot(x) - b) ** 2) + alpha * np.linalg.norm(z, 1)


def shrinkage(x, kappa):
    return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)


def lu_factor(A, rho):
    m, n = A.shape
    if m >= n:  # if skinny
        L = np.linalg.cholesky(A.T.dot(A) + rho * np.eye(n))
    else:  # if fat
        L = np.linalg.cholesky(np.eye(m) + 1. / rho * A.dot(A.T))

    U = L.T
    return L, U
