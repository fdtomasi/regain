# BSD 3-Clause License

# Copyright (c) 2019, regain authors
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
import numpy as np

from .group_lasso_overlap_ import objective


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
    QUIET = 0
    MAX_ITER = 100
    RELTOL = 1e-2
    ABSTOL = 1e-4

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
        Ai = A[:, i * ni : (i + 1) * ni]
        Di, Vi = np.linalg.eig(Ai.T.dot(Ai))
        Vis.append(Vi)
        Dis.append(Di)
        Ats.append(Ai.T)

    hist = []
    for k in range(MAX_ITER):
        # % x-update (to be done in parallel)
        for i in range(N):
            Ai = A[:, i * ni : (i + 1) * ni]
            xx = x_update(Ai, Aixi[:, i] + z - Axbar - u, lamda / rho, Vis[i], Dis[i])
            x[:, i] = xx
            Aixi[:, i] = Ai.dot(x[:, i])

        # % z-update
        zold = z
        Axbar = 1.0 / N * A.dot(x.ravel())

        Axbar_hat = alpha * Axbar + (1 - alpha) * zold
        z = (b + rho * (Axbar_hat + u)) / (N + rho)

        # % u-update
        u = u + Axbar_hat - z

        # % compute the dual residual norm square
        s = 0
        q = 0
        zsold = zs
        zs = z.reshape(-1, 1).dot(np.ones((1, N)))
        zs += Aixi - Axbar.reshape(-1, 1).dot(np.ones((1, N)))
        for i in range(N):
            # % dual residual norm square
            s = s + np.linalg.norm(-rho * Ats[i].dot((zs[:, i] - zsold[:, i]))) ** 2
            # % dual residual epsilon
            q = q + np.linalg.norm(rho * Ats[i].dot(u)) ** 2

        # % diagnostics, reporting, termination checks
        history = []
        history.append(objective(A, b, lamda, x, z))
        history.append(np.sqrt(N) * np.linalg.norm(z - Axbar))
        history.append(np.sqrt(s))

        history.append(np.sqrt(n) * ABSTOL + RELTOL * max(np.linalg.norm(Aixi, "fro"), np.linalg.norm(-zs, "fro")))
        history.append(np.sqrt(n) * ABSTOL + RELTOL * np.sqrt(q))

        hist.append(history)
        if history[1] < history[3] and history[2] < history[4]:
            break

    return x


def x_update(A, b, kappa, V, D):
    _, n = A.shape
    q = A.T.dot(b)

    x = np.zeros(n)
    if np.linalg.norm(q) > kappa:
        # bisection on t
        lower = 0
        upper = 1e10
        for _ in range(100):
            t = (upper + lower) / 2.0

            x = V.dot(V.T.dot(q) / (D + t))
            if t > kappa / np.linalg.norm(x):
                upper = t
            else:
                lower = t
            if upper - lower <= 1e-6:
                break
    return x
