import numpy as np

def group_lasso_feat_split(A, b, lamda=1.0, ni=2, rho=1.0, alpha=1.0):
    # % group_lasso_feat_split  Solve group lasso problem via ADMM feature splitting
    # %
    # % [x, history] = group_lasso_feat_split(A, b, p, lambda, rho, alpha);
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
    # % This version is a (serially) distributed, feature splitting example.
    # %
    # %
    # % More information can be found in the paper linked at:
    # % http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    # %
    #
    # t_start = tic;
    # Global constants and defaults
    #
    QUIET    = 0;
    MAX_ITER = 100;
    RELTOL  = 1e-2;
    ABSTOL   = 1e-4;

    m, n = A.shape

    # % check that ni divides in to n
    # if (rem(n,ni) ~= 0)
    #     error('invalid block size');
    # end
    # % number of subsystems
    N = n / ni

    # rho = RHO;
    # alpha = ALPHA;    % over-relaxation parameter

    x = np.zeros((ni, N))
    z = np.zeros(m)
    u = np.zeros(m)
    Axbar = np.zeros(m)

    zs = np.zeros((m, N))
    Aixi = np.zeros((m, N))

    Vis, Dis = [], []
    Ats = []
    # % pre-factor
    for i in range(N):
        Ai = A[:, i * ni:(i + 1) * ni]
        Di, Vi = np.linalg.eig(Ai.T.dot(Ai))
        Vis.append(Vi)
        Dis.append(Di)
        Ats.append(Ai.T)

    hist = []
    for k in range(MAX_ITER):
        # % x-update (to be done in parallel)
        for i in range(N):
            Ai = A[:, i * ni:(i + 1) * ni]
            xx = x_update(Ai, Aixi[:, i] + z - Axbar - u, lamda / rho, Vis[i], Dis[i])
            x[:, i] = xx
            Aixi[:, i] = Ai.dot(x[:, i])

        # % z-update
        zold = z
        Axbar = 1. / N * A.dot(x.ravel())

        Axbar_hat = alpha * Axbar + (1 - alpha) * zold
        z = (b + rho*(Axbar_hat + u))/(N+rho);

        # % u-update
        u = u + Axbar_hat - z

        # % compute the dual residual norm square
        s = 0
        q = 0
        zsold = zs
        zs = z.reshape(-1,1).dot(np.ones((1, N)))
        zs += Aixi - Axbar.reshape(-1,1).dot(np.ones((1, N)))
        for i in range(N):
            # % dual residual norm square
            s = s + np.linalg.norm(-rho * Ats[i].dot((zs[:,i] - zsold[:,i])))**2
            # % dual residual epsilon
            q = q + np.linalg.norm(rho*Ats[i].dot(u))**2;

        # % diagnostics, reporting, termination checks
        history = []
        history.append(objective(A, b, lamda, N, x, z))
        history.append(np.sqrt(N)*np.linalg.norm(z - Axbar))
        history.append(np.sqrt(s))

        history.append(np.sqrt(n)*ABSTOL + RELTOL*max(np.linalg.norm(Aixi,'fro'), np.linalg.norm(-zs, 'fro')))
        history.append(np.sqrt(n)*ABSTOL + RELTOL*np.sqrt(q))

        hist.append(history)
        if history[1] < history[3] and history[2] < history[4]:
            break

    return x

def objective(A, b, alpha, N, x, z):
    return .5 * np.sum((N * z - b)**2) + alpha * np.sum(np.linalg.norm(x))

def x_update(A, b, kappa, V, D):
    m, n = A.shape
    q = A.T.dot(b)

    if (np.linalg.norm(q) <= kappa):
        x = np.zeros(n)
    else:
        # % bisection on t
        lower = 0
        upper = 1e10
        for i in range(100):
            t = (upper + lower) / 2.

            x = V.dot(V.T.dot(q) / (D + t))
            if t > kappa / np.linalg.norm(x):
                upper = t
            else:
                lower = t
            if (upper - lower <= 1e-6):
                break
    return x
