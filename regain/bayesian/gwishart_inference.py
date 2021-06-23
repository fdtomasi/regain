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
"""Bayesian inference of a sparse inverse covariance matrix.

Based on
https://github.com/probml/pmtksupport/blob/master/GGM-GWishart/GWishartScore.m
"""
from functools import partial

import numpy as np
from numpy import binary_repr
from scipy import linalg
from scipy.optimize import minimize
from scipy.special import comb
from sklearn.covariance import empirical_covariance
from sklearn.covariance.empirical_covariance_ import log_likelihood
from sklearn.linear_model import LassoLars
from sklearn.utils import Bunch, check_array
from sklearn.utils.extmath import fast_logdet

from regain.covariance.graphical_lasso_ import GraphicalLasso, graphical_lasso


def mk_all_ugs(n_dim):
    """Utility for generating all possible graphs."""
    nedges = int(comb(n_dim, 2))
    m = 2 ** nedges

    ind = np.array([list(binary_repr(x, width=len(binary_repr(m - 1)))) for x in range(m)]).astype(int)
    ord = np.argsort(ind.sum(axis=1))
    ind = ind[ord]

    ut = np.triu(np.ones((n_dim, n_dim)), 1) > 0

    Gs = []
    for i in range(m):
        G = np.zeros((n_dim, n_dim))
        G[ut] = ind[i]
        G = (G + G.T) > 0
        Gs.append(G)
    return Gs


def markov_blankets(graphs, boolean=True, tol=1e-8, unique=False):
    """For each variable, list each of its markov blankets in graphs."""
    m_blankets = [np.array([G[i] for G in graphs]) for i in range(graphs[0].shape[0])]
    for i, mb in enumerate(m_blankets):
        mb[:, i] = 0
        mb[np.abs(mb) < tol] = 0
    if boolean:
        m_blankets = [mb != 0 for mb in m_blankets]
    if unique:
        # discard same blankets
        # works with numpy >= 1.13.0
        m_blankets = np.unique(np.array(m_blankets), axis=1)
    return m_blankets


def score_blankets(blankets, X, alphas=(0.01, 0.5, 1)):
    """Score the markov blankets for node i.

    Restrict data to markov blanket, use as y (the variable to regress)
    the node i.
    """
    scores_all = []
    for i, mb_i in enumerate(blankets):
        scores = []
        for j, mb in enumerate(mb_i):
            X_mb = X[:, mb]
            if X_mb.shape[1] < 1:
                X_mb = np.zeros((X.shape[0], 1))

            y_mb = X[:, i]
            score = np.sum([LassoLars(alpha=alpha).fit(X_mb, y_mb).score(X_mb, y_mb) for alpha in alphas])

            scores.append(score)
        scores_all.append(scores)

    ss = np.exp(np.array(scores_all))
    normalized_scores = ss / np.repeat(ss.sum(axis=1)[:, None], ss.shape[1], axis=1)
    return normalized_scores


def _get_graphs(blankets, scores, n_dim, n_resampling=200):
    idx = np.array([np.random.choice(scores.shape[1], p=scores[i], size=n_resampling) for i in range(n_dim)])

    graphs_ = np.array([blankets[i][idx[i]] for i in range(n_dim)]).transpose(1, 0, 2)
    # symmetrise with AND operator -> product
    graphs = np.array([g * g.T for g in graphs_])
    return graphs


def covsel(x, p, nonZero, C):
    """Objective and gradient for MLE of precision given empirical covariance.

    nonZero is a list of non-zero upper triangle precision matrix entries.
    Based on sparse GGM estimation code by Mark Schmidt.
    """
    X = np.zeros((p, p))
    X[nonZero] = x  # fill the diagonal and upper triangle
    X += np.triu(X, 1).T  # fill the lower triangle

    # Fast Way to compute -logdet(X) + tr(X*C)
    # f = -2*sum(log(diag(R))) + sum(sum(C.*X)) + (lambda/2)*sum(X(:).^2);
    f = -fast_logdet(X) + np.sum(C * X)

    if f < np.inf:
        g = C - linalg.pinvh(X)
        g += np.tril(g, -1).T  # add contribution from lower to upper triangle
        g = g[nonZero]
    else:
        g = 0
    return f, g


def precision_selection(G, n_dim, C):
    """MLE of precision matrix given non-zero entries in G."""
    function = partial(covsel, p=n_dim, nonZero=G, C=C)

    x0 = np.eye(n_dim)
    min_obj = minimize(function, x0[G], jac=True)
    x = min_obj.x

    XX = np.zeros((n_dim, n_dim))
    XX[G] = x
    XX += np.triu(XX, 1).T
    return XX


def GWishartFit(X, G, GWprior, mode="covsel"):
    """Fit G-Wishart distribution."""
    n_samples, n_dim = X.shape

    d0 = GWprior.d0
    S0 = GWprior.S0

    # check prior size violations
    if G.shape[0] != n_dim or G.shape[1] != n_dim:
        raise ValueError("G must be p-by-p, with p dimensions X")
    if S0.shape[0] != n_dim or S0.shape[1] != n_dim:
        raise ValueError("GWprior.S0 must be p-by-p, with p dimensions X")

    # compute posterior scatter matrix
    dn = n_samples + d0

    # X'*X - but I dont assume X to be centered
    emp_cov = empirical_covariance(X)
    S = n_samples * emp_cov
    C = (S + S0) / (dn - 2)

    if mode == "covsel":
        precision = precision_selection(G, n_dim, C)
    else:
        # use graph_lasso
        # convert G to alpha
        alpha = np.zeros_like(G, dtype=float)
        alpha[~(G + G.T)] = np.inf
        precision = graphical_lasso(emp_cov=C, alpha=alpha)[0]

    return precision, S


def compute_score(X, G, P, S, GWprior=None, score_method="bic"):
    """Compute score function of P."""
    n_samples, n_dim = X.shape

    d0 = GWprior.d0
    S0 = GWprior.S0

    # check prior size violations
    if S0.shape[0] != n_dim or S0.shape[1] != n_dim:
        raise ValueError("GWprior.S0 must be p-by-p, with p dimensions X")

    dn = n_samples + d0
    # C = (S + S0) / (dn - 2)

    # % need logdetP and invP
    # es, Q = np.linalg.eigh(x)
    # Inv = np.linalg.multi_dot((Q, np.diag(1. / es), Q.T))
    U, s, Vh = linalg.svd(P)

    # check
    invP = np.linalg.multi_dot((Vh.T, np.diag(1.0 / s), U.T))
    logdetP = np.sum(np.log(s))

    # % compute loglik
    loglik = n_samples * log_likelihood(S / n_samples, P)

    num_edges = np.triu(G, 1).sum()
    dof = num_edges + n_dim

    # pcor = cov2cor(P);

    # % the posterior Sn parameter
    # Sn = (dn - 2) * invP
    logh = (dn - 2) / 2.0 * (n_dim + logdetP)

    # find full param set V
    Vi, Vj = np.nonzero(np.triu(G))

    # to be the same as matlab
    idx = np.argsort(Vj)
    Vi, Vj = Vi[idx], Vj[idx]

    GWpost = Bunch()
    GWpost.Sn = S + S0
    # GWpost.C = C
    GWpost.dn = dn
    GWpost.P = P
    GWpost.num_edges = num_edges
    GWpost.dof = dof
    GWpost.logdetP = logdetP
    GWpost.loglik = loglik

    if score_method == "bic":
        score = loglik - dof * np.log(n_samples) / 2 if n_samples > 0 else 0

    elif score_method == "diaglaplace":
        # Diagonal hessian laplace approximation
        diagH = np.zeros(dof)
        for e1 in range(dof):
            # e2 = e1

            M1 = np.zeros((n_dim, n_dim))
            # M2 = M1.copy()

            nz1 = [Vi[e1], Vj[e1]]
            # nz2 = [Vi[e2], Vj[e2]]
            M1[:, nz1] = invP[:, [Vj[e1], Vi[e1]]]
            # M2[:, nz2] = invP[:, [Vj[e2], Vi[e2]]]

            # A = M1[nz2][:, nz1]
            # B = M2[nz1][:, nz2]
            A = M1[nz1][:, nz1]
            B = A

            tmp2 = A[0, :].dot(B[:, 0]) + A[1, :].dot(B[:, 1])

            diagH[e1] = -(dn - 2) * tmp2 / 2
            # diagH(e1) = -(dn-2) * trace(M1(nz2,nz1)*M2(nz1,nz2))/2;

        logdetHdiag = sum(np.log(-diagH))
        lognormconst = dof * np.log(2 * np.pi) / 2 + logh - logdetHdiag / 2.0
        score = lognormconst - GWprior.lognormconst - n_samples * n_dim * np.log(2 * np.pi) / 2
        GWpost.lognormconst = lognormconst

    elif score_method == "laplace":
        # Full laplace approximation
        H = np.empty((dof, dof))
        for e1 in range(dof):
            # nz1 = [Vi[e1], Vj[e1]]
            i, j = Vi[e1], Vj[e1]

            for e2 in range(e1, dof):
                # nz2 = [Vi[e2], Vj[e2]]
                l, m = Vi[e2], Vj[e2]
                # A = invP[nz2][:, [Vj[e1], Vi[e1]]]
                # B = invP[nz1][:, [Vj[e2], Vi[e2]]]
                A = invP[[l, m]][:, [j, i]]
                B = invP[[i, j]][:, [m, l]]

                # tmp2 = A[0, :].dot(B[:, 0]) + A[1, :].dot(B[:, 1])
                # tmp2 = np.trace(A.dot(B))
                tmp2 = (A * B.T).sum()
                H[e2, e1] = H[e1, e2] = -(dn - 2) * tmp2 / 2.0

        # neg Hessian will be posdef
        logdetH = 2 * sum(np.log(np.diag(linalg.cholesky(-H))))
        lognormconst = dof * np.log(2 * np.pi) / 2 + logh - logdetH / 2.0
        score = lognormconst - GWprior.lognormconst - n_samples * n_dim * np.log(2 * np.pi) / 2
        GWpost.lognormconst = lognormconst

    GWpost.score = score
    return GWpost


def GWishartScore(X, G, d0=3, S0=None, score_method="bic", mode="covsel"):
    """Compute score of G-Wishart distribution."""
    # %Initialize GW prior structure
    n_samples, n_dim = X.shape
    if S0 is None:
        S0 = (d0 - 2) * np.diag(np.diag(np.cov(X, bias=True)))

    GWprior = Bunch(d0=d0, S0=S0, lognormconst=0, lognormconstDiag=0)

    # %If method is laplace, compute laplace approximation to denominator
    if score_method in ["laplace", "diaglaplace"]:
        noData = np.zeros((0, n_dim))

        P0, S_noData = GWishartFit(noData, G, GWprior)
        GWtemp = compute_score(noData, G, P0, S_noData, GWprior=GWprior, score_method=score_method)
        GWprior.lognormconst = GWtemp.lognormconst

    # Compute the map precision matrix P
    P, S = GWishartFit(X, G, GWprior, mode=mode)

    # Compute the score
    GWpost = compute_score(X, G, P, S, GWprior, score_method)
    return GWpost


def bayesian_graphical_lasso(
    X,
    tol=1e-8,
    alphas=None,
    n_resampling=200,
    mode="gl",
    top_n=1,
    scoring="diaglaplace",
    assume_centered=False,
    max_iter=100,
):
    """Bayesian graphical lasso."""
    n_samples, n_dim = X.shape

    if alphas is None:
        alphas = np.logspace(-2, 0, 20)

    # get a series of Markov blankets for vaiours alphas
    mdl = GraphicalLasso(assume_centered=assume_centered, tol=tol, max_iter=max_iter, verbose=False)
    precisions = [mdl.set_params(alpha=a).fit(X).precision_ for a in alphas]
    mblankets = markov_blankets(precisions, tol=tol, unique=True)

    normalized_scores = score_blankets(mblankets, X=X, alphas=[0.01, 0.5, 1])

    graphs = _get_graphs(mblankets, normalized_scores, n_dim=n_dim, n_resampling=n_resampling)

    nonzeros_all = [np.triu(g, 1) + np.eye(n_dim, dtype=bool) for g in graphs]

    # Roverato'02: convert from HIW to G-Wishart (delta + |V| - 1)
    d0 = 3 + n_dim - 1
    S0 = np.eye(n_dim)  # same as Roverato'02

    # Find non-zero elements of upper triangle of G
    # make sure diagonal is non-zero
    # G = nonzeros_all[1] # probably can discard if all zeros?
    res = [GWishartScore(X, G, d0=d0, S0=S0, mode=mode, score_method=scoring) for G in nonzeros_all]

    top_n = [x.P for x in sorted(res, key=lambda x: x.score)[::-1][:top_n]]
    return np.mean(top_n, axis=0)


class BayesianGraphicalLasso(GraphicalLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Parameters
    ----------
    alpha : positive float, default 0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

    rho : positive float, default 1
        Augmented Lagrangian parameter.

    over_relax : positive float, deafult 1
        Over-relaxation parameter (typically between 1.0 and 1.8).

    tol : positive float, default 1e-4
        Absolute tolerance to declare convergence.

    rtol : positive float, default 1e-4
        Relative tolerance to declare convergence.

    max_iter : integer, default 100
        The maximum number of iterations.

    verbose : boolean, default False
        If verbose is True, the objective function, rnorm and snorm are
        printed at each iteration.

    assume_centered : boolean, default False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    update_rho_options : dict, default None
        Options for the update of rho. See `update_rho` function for details.

    compute_objective : boolean, default True
        Choose if compute the objective function during iterations
        (only useful if `verbose=True`).

    mode : {'admm'}, default 'admm'
        Minimisation algorithm. At the moment, only 'admm' is available,
        so this is ignored.

    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(
        self,
        alpha=0.01,
        max_iter=100,
        alphas=None,
        n_resampling=200,
        tol=1e-4,
        verbose=False,
        assume_centered=False,
        mode="gl",
        scoring="diaglaplace",
        top_n=1,
    ):
        super(GraphicalLasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter, verbose=verbose, assume_centered=assume_centered, mode=mode
        )
        self.alphas = alphas
        self.n_resampling = n_resampling
        self.scoring = scoring
        self.top_n = top_n

    def fit(self, X, y=None):
        """Fit the GraphLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : (ignored)

        """
        # Covariance does not make sense for a single feature
        X = check_array(X, ensure_min_features=2, ensure_min_samples=2, estimator=self)

        if self.alphas is None:
            self.alphas = np.logspace(-2, 0, 20)

        self.precision_ = bayesian_graphical_lasso(
            X,
            tol=self.tol,
            alphas=self.alphas,
            n_resampling=self.n_resampling,
            mode=self.mode,
            scoring=self.scoring,
            assume_centered=self.assume_centered,
            max_iter=self.max_iter,
            top_n=self.top_n,
        )
        return self
