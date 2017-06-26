import numpy as np
from rgi.admm_lasso import lu_factor

def group_lasso(A, b, lamda=1.0, p=None, rho=1.0, alpha=1.0, max_iter=1000, tol=0.0001):
    # % group_lasso  Solve group lasso problem via ADMM
    # %
    # % [x, history] = group_lasso(A, b, p, lambda, rho, alpha);
    # %
    # % solves the following problem via ADMM:
    # %
    # %   minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))
    # %
    # % The input p is a K-element vector giving the block sizes n_i, so that x_i
    # % is in R^{n_i}.
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

    RELTOL = 1e-2

    m, n = A.shape

    # % save a matrix-vector multiply
    Atb = A.T.dot(b)
    # % check that sum(p) = total number of elements in x
    # if (sum(p) ~= n)
    #     error('invalid partition');
    # end

    # % cumulative partition
    # cum_part = np.cumsum(p)

    # ADMM solver
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    # % pre-factor
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

        # % z-update
        zold = z
        x_hat = alpha * x + (1 - alpha) * zold
        for group in p:
            z[group] = shrinkage(x_hat[group] + u[group], lamda / rho)
        u += (x_hat - z)

        # % diagnostics, reporting, termination checks
        history = []
        history.append(objective(A, b, lamda, p, x, z))  # obj

        history.append(np.linalg.norm(x - z))  # r norm
        history.append(np.linalg.norm(-rho*(z - zold)))  # s norm

        history.append(np.sqrt(n)*tol + RELTOL*max(np.linalg.norm(x), np.linalg.norm(-z)))  # eps pri
        history.append(np.sqrt(n)*tol + RELTOL*np.linalg.norm(rho*u))  # eps dual

        hist.append(history)
        if history[1] < history[3] and history[2] < history[4]:
            break

    return z


def objective(A, b, alpha, groups, x, z):
    # obj = 0
    # for i, group in enumerate(p):
    #     obj = obj + np.linalg.norm(z[group])
    penalty = np.sum([np.linalg.norm(z[g]) for g in groups])
    return .5 * np.sum((A.dot(x) - b) ** 2) + alpha * penalty


def shrinkage(x, kappa):
    return np.maximum(0., 1 - kappa / np.linalg.norm(x)) * x
