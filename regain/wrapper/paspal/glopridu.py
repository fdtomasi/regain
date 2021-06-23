"""PASPAL implementation (in Python, based on the original MATLAB).

Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco (Matlab implementation)
Copyright 2018 Federico Tomasi (Python implementation)

"""
from __future__ import print_function, division

from itertools import combinations

import numpy as np
from scipy import linalg


def glo_prox(w0, tau, blocks, weights, lamda0, tol, max_iter, verbose=0, beta=0.5, sigma=0.1, s_beta=1, epsilon=1e-3):
    """Proximity operator of the group lasso with overlap penalty.

    First, identifies "active" blocks, then apply Bersekas's
    projected Newton method on the dual space.

    """
    d = w0.size
    B = len(blocks)

    # ------------NEW
    # weights = zeros(B,1);
    # for g = 1:B;
    #     weights(g) = length(blocks{g})*tau^2; %weight is sqrt{|g|}
    #     weights(g) = tau^2; %not weighted
    # end
    # ------------NEW
    weights = (weights * tau) ** 2

    # if lamda is not initialized, then initialize it to 0
    if not lamda0:
        lamda0 = np.zeros(B)

    lamda_tot = np.zeros(B)

    # Identify active blocks, by removing blocks such that w0 is already
    # inside the corresponding cylinder
    # to_be_projected = np.zeros(B, dtype=bool)
    # for g in range(B):
    #     to_be_projected[g] = np.linalg.norm(w0[blocks[g]], 2) >= weights[g]
    norm_blocks = np.array([np.linalg.norm(w0[block], 2) for block in blocks])
    to_be_projected = norm_blocks >= weights

    weights = weights[to_be_projected]
    blocks = blocks[to_be_projected]
    lamda = lamda0[to_be_projected]

    B = len(blocks)
    if B == 0:
        return np.zeros(d), 0, lamda_tot

    I = np.zeros((d, B))
    for g in range(B):
        I[blocks[g], g] = 1

    # Bersekas constrained Newton method
    i_null = 0
    iteration_ = 0
    for iteration_ in range(max_iter):
        if verbose > 1:
            print(".... interior, it %d out of %d" % (iteration_, max_iter))
        lamda_prev = lamda.copy()
        s = I.dot(lamda_prev)
        denominator = 1.0 / (1 + s)
        grad = weights - I.T.dot((w0 * denominator) ** 2)
        epsk = min(epsilon, np.linalg.norm(lamda - np.maximum(0, lamda - grad)))
        tmp = 2 * w0 ** 2 * denominator ** 3

        I_inactive = np.where(np.logical_or(grad <= 0, lamda > epsk))[0]
        n_inactive = I_inactive.size
        B_inactive = np.zeros((n_inactive, n_inactive))

        # for g, g_inactive in enumerate(I_inactive):
        #     B_inactive[g, g] = np.sum(tmp[blocks[g_inactive]])
        #     for k in range(g + 1, n_inactive):
        #         B_inactive[k, g] = B_inactive[g, k] = np.sum(tmp[
        #             np.intersect1d(blocks[g_inactive],
        #                            blocks[I_inactive[k]])])
        # import time
        # tic = time.time()

        blocks_inactive_set = [set(blocks[c]) for c in I_inactive]
        B_inactive[np.triu_indices_from(B_inactive, 1)] = [
            tmp[list(x.intersection(y))].sum() for x, y in combinations(blocks_inactive_set, 2)
        ]
        B_inactive += B_inactive.T
        B_inactive.flat[:: n_inactive + 1] = [tmp[list(x)].sum() for x in blocks_inactive_set]

        # print("time 1: ", time.time() - tic)
        # tic = time.time()
        #
        # blocks_inactive = [blocks[c] for c in I_inactive]
        # C_inactive[np.triu_indices_from(C_inactive, 1)] = [
        #     tmp[np.intersect1d(x, y)].sum()
        #     for x, y in combinations(blocks_inactive, 2)]
        # C_inactive += C_inactive.T
        # C_inactive.flat[::n_inactive + 1] = [
        #     tmp[x].sum() for x in blocks_inactive]
        #
        # print("time 2: ", time.time() - tic)
        # assert np.allclose(B_inactive, C_inactive)

        p_inactive = linalg.pinv(B_inactive).dot(grad[I_inactive])

        I_active = np.where(np.logical_and(grad > 0, lamda <= epsk))[0]
        # n_active = I_active.size
        if n_inactive == 0:
            i_null += 1
        else:
            i_null = 0

        if i_null >= 5:
            break

        # B_active = np.empty(n_active)
        B_active = np.array([tmp[blocks[g_active]].sum() for g_active in I_active])

        p_active = grad[I_active] / B_active

        x_inactive = grad[I_inactive].T.dot(p_inactive)
        test = 1
        m = 0
        lamda_m = np.zeros(B)
        while test:
            m += 1
            step = beta ** m * s_beta
            lamda_m[I_active] = np.maximum(0, lamda_prev[I_active] - step * p_active)
            lamda_m[I_inactive] = np.maximum(0, lamda_prev[I_inactive] - step * p_inactive)
            s_m = I.dot(lamda_m)
            fdiff = (w0 ** 2).T.dot(I.dot(lamda_m - lamda) / ((1 + s_m) * (1 + s))) + np.sum(
                weights * (lamda - lamda_m)
            )
            x_active = grad[I_active].T.dot(lamda[I_active] - lamda_m[I_active])
            test = fdiff < sigma * (step * x_inactive + x_active)

        lamda = lamda_m.copy()
        if all(grad[lamda == 0] >= 0) and all(abs(grad[lamda > 0]) < tol):
            break

    # given the solution of dual problem, lamda, compute the primal solution w
    s = I.dot(lamda)
    w = w0 * (1 - 1.0 / (1 + s))
    lamda_tot[to_be_projected] = lamda

    return w, iteration_, lamda_tot


def glopridu_algorithm(
    X,
    Y,
    blocks,
    tau,
    weights=None,
    smooth_par=0,
    beta0=None,
    lamda0=None,
    sigma0=None,
    max_iter_ext=1e4,
    max_iter_int=100,
    tol_ext=1e-6,
    tol_int=1e-4,
    verbose=0,
):
    """Add Doc."""
    n, d = X.shape
    blocks = np.array(blocks)

    # if only the number of blocks is given, build blocks of equal
    # cardinality with sequential features
    # if ~iscell(blocks);
    #     blocks = num2cell(reshape(1:d,d/blocks,blocks),1);
    # end

    # if sigma is not specified in input, set  it to as a/n
    if not sigma0:
        sigma0 = np.linalg.norm(X, 2) ** 2 / n  # step size for smooth_par=0

    # if weights are not specified in input, set them to 1
    if not weights:
        weights = np.ones(len(blocks))
        # weights = np.ones(d)

    mu = smooth_par * sigma0  # smoothness parameter is rescaled
    sigma = sigma0 + mu  # step size

    # useful normalization that avoid computing the same computations for
    # each iteration
    mu_s = mu / sigma
    tau_s = tau / sigma
    XT = X.T / (n * sigma)

    # initialization
    if not beta0:
        beta0 = np.zeros(d)

    # initialization for iterate n_iter-1
    beta = beta0.copy()
    # initialization for combination of the previous 2 iterates
    # (iteratations n_iter_1 and n_iter-2)
    h = beta0.copy()
    # initialization for the adaptive parameter used to combine
    # the previous 2 iterates when building h
    t = 1
    # precomputes X*beta and X*h to avoid computing them twice
    Xb = X.dot(beta)
    Xh = Xb.copy()
    lamda = np.zeros(len(blocks)) if lamda0 is None else lamda0

    # initialization for the dual vector in computing the projection
    # lamda_prev = lamda.copy()
    n_iter = 0
    for n_iter in range(1, int(max_iter_ext) + 1):
        if verbose > 0:
            print("Iteration %d/%d" % (n_iter - 1, max_iter_ext))

        beta_prev = beta.copy()
        Xb_prev = Xb.copy()

        # computes the gradient step
        # beta_noproj = h.*(1-mu_s) + XT*(Y-Xh);

        # compute the proximity operator with tolerance depending on k
        w0 = h * (1.0 - mu_s) + XT.dot(Y - Xh)
        beta, _, lamda = glo_prox(
            w0=w0,
            tau=tau_s,
            blocks=blocks,
            weights=weights,
            lamda0=lamda,
            tol=tol_int * n_iter ** (-3.0 / 2),
            max_iter=max_iter_int,
            verbose=verbose,
        )

        # lamda_prev = lamda.copy()
        Xb = X.dot(beta)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))

        # combination of the 2 previous iterates
        h = beta + (t - 1) / t_new * (beta - beta_prev)
        Xh = Xb * (1.0 + (t - 1.0) / t_new) + (1.0 - t) / t_new * Xb_prev
        t = t_new

        if np.linalg.norm(Xb - Xb_prev) <= np.linalg.norm(Xb_prev) * tol_ext and n_iter > 1:
            break

    return beta, lamda, n_iter
