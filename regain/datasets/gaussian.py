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
from __future__ import division

import warnings

import numpy as np
from scipy import signal
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.utils import check_random_state

from regain.utils import ensure_posdef
from regain.utils import is_pos_def
from regain.utils import is_pos_semidef
from regain.utils import normalize_matrix


def _compute_probabilities(locations, random_state, mask=None):
    p = len(locations)
    if mask is None:
        mask = np.zeros((p, p))
    probabilities = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            d = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
            d = d / 10.0 if mask[i, j] else d
            probabilities[i, j] = 2 * norm.pdf(d * np.sqrt(p), scale=1)
    thresholds = random_state.uniform(low=0, high=1, size=(p, p))
    probabilities[np.where(thresholds > probabilities)] = 0
    return probabilities


def _permute_locations(locations, random_state):
    new_locations = []
    for l in locations:
        perm = random_state.uniform(low=-0.03, high=0.03, size=(1, 2))
        new_locations.append(np.maximum(np.array(l) + np.array(perm), 0).flatten().tolist())
    return new_locations


def _theta_my(probs, p, random_state, value=0.245):
    theta = np.zeros((p, p))
    for i in range(p):
        ix = np.argsort(probs[i, :])[::-1]
        no_nonzero = np.sum((theta != 0).astype(int), axis=1)[i]
        ixs = []
        sel = 1
        j = 0
        while sel <= 4 - no_nonzero and j < p:
            if ix[j] < i:
                j += 1
                continue
            ixs.append(ix[j])
            sel += 1
            j += 1
        theta[i, ixs] = theta[ixs, i] = value
    to_remove = np.where(np.sum((theta != 0).astype(int), axis=1) > 4)[0]
    for t in to_remove:
        edges = np.nonzero(theta[t, :])[0]
        random_state.shuffle(edges)
        removable = edges[:-4]
        theta[t, removable] = theta[removable, t] = 0

    np.fill_diagonal(theta, 1)
    assert is_pos_def(theta)
    return theta


def data_Meinshausen_Yuan(p=198, h=2, n=200, T=10, random_state=None, **kwargs):
    random_state = check_random_state(random_state)
    nodes_locations = [list(random_state.uniform(size=2)) for i in range(p)]
    probs = _compute_probabilities(nodes_locations, random_state)
    theta = _theta_my(probs, p, random_state=random_state)

    latent = np.zeros((h, p))
    for i in range(h):
        for j in range(p):
            latent[i, j] = random_state.uniform(low=0, high=0.01, size=1)

    L = latent.T.dot(latent)
    theta_observed = theta - L
    assert is_pos_def(theta_observed)
    sigma = np.linalg.inv(theta_observed)
    assert is_pos_def(sigma)

    samples = random_state.multivariate_normal(np.zeros(p), sigma, size=n)

    locations = [nodes_locations]
    thetas = [theta]
    thetas_observed = [theta_observed]
    covariances = [sigma]
    sampless = [samples]
    for t in range(T - 1):
        theta_prev = thetas[t]
        locs_prev = locations[t]
        locs = _permute_locations(locs_prev, random_state)

        locations.append(locs)
        probs = _compute_probabilities(locs, random_state, theta_prev != 0)

        theta = _theta_my(probs, p, random_state=random_state)
        thetas.append(theta)
        theta_observed = theta - L
        assert is_pos_def(theta_observed)
        sigma = np.linalg.inv(theta_observed)
        assert is_pos_def(sigma)
        thetas_observed.append(theta_observed)
        covariances.append(sigma)
        sampless.append(random_state.multivariate_normal(np.zeros(p), sigma, size=n))

    return locations, thetas, thetas_observed, covariances, latent, sampless


def data_Meinshausen_Yuan_sparse_latent(p=198, h=2, n=200, T=10, random_state=None, **kwargs):
    random_state = check_random_state(random_state)
    nodes_locations = [list(random_state.uniform(size=2)) for i in range(p + h)]
    probs = _compute_probabilities(nodes_locations, random_state)
    theta = _theta_my(probs[:, :-h], p, random_state=random_state)

    latent = np.zeros((h, p + h))
    to_put = (p + h) * 0.5
    value = 0.98 / to_put
    for i in range(h):
        pb = probs[p + i, :]
        ix = np.argsort(pb)[::-1]

        no_nonzero = np.sum((latent != 0).astype(int), axis=1)[i]
        ixs = []
        sel = 1
        j = 0
        while sel <= min(to_put - no_nonzero, np.nonzero(pb)[0].shape[0]) and j < h + h:
            if ix[j] < i:
                j += 1
                continue
            ixs.append(ix[j])
            sel += 1
            j += 1
        latent[i, ixs] = value
        for ix in ixs:
            if ix < h:
                latent[ix, i] = value
    to_remove = np.where(np.sum((theta != 0).astype(int), axis=1) > to_put)[0]
    for t in to_remove:
        edges = np.nonzero(theta[t, :])[0]
        random_state.shuffle(edges)
        removable = edges[:-to_put]
        latent[t, removable] = 0
        latent[removable, t] = 0
    for i in range(h):
        latent[i, i] = 1
    assert is_pos_def(latent[:h, :h])
    connections = latent[:, h:]

    theta_H = latent[:h, :h]
    theta_obs = theta - connections.T.dot(np.linalg.inv(theta_H)).dot(connections)
    assert is_pos_def(theta_obs)
    sigma = np.linalg.inv(theta_obs)
    assert is_pos_def(sigma)

    samples = random_state.multivariate_normal(np.zeros(p), sigma, size=n)

    thetas_O = [theta]
    thetas_H = [theta_H]
    thetas_obs = [theta_obs]
    locations = [nodes_locations]
    sigmas = [sigma]
    sampless = [samples]
    for t in range(T - 1):
        theta_prev = thetas_O[t]
        lat_prev = thetas_H[t]
        locs_prev = locations[t]
        locs = _permute_locations(locs_prev, random_state)

        locations.append(locs)
        probs_theta = _compute_probabilities(locs[:p], random_state, theta_prev != 0)
        probs_lat = _compute_probabilities(locs[-h:], random_state, lat_prev != 0)

        theta = _theta_my(probs_theta, p, random_state=random_state)
        thetas_O.append(theta)

        lat = np.zeros((h, h))
        to_put = [np.nonzero(lat_prev[i, :])[0].shape[0] for i in range(h)]

        for i in range(h):
            ix = np.argsort(probs_lat[i, :])[::-1]
            no_nonzero = np.sum((lat != 0).astype(int), axis=1)[i]
            ixs = []
            sel = 1
            j = 0
            while sel <= to_put[i] - no_nonzero and j < h:
                if ix[j] < i:
                    j += 1
                    continue
                ixs.append(ix[j])
                sel += 1
                j += 1
            lat[i, ixs] = value
            lat[ixs, i] = value
        to_remove = np.where(np.sum((lat != 0).astype(int), axis=1) > to_put)[0]
        for t in to_remove:
            edges = np.nonzero(lat[t, :])[0]
            random_state.shuffle(edges)
            removable = edges[:-4]
            lat[t, removable] = 0
            lat[removable, t] = 0
        np.fill_diagonal(lat, 1)
        assert is_pos_def(lat)
        thetas_H.append(lat)

        theta_obs = theta - connections.T.dot(np.linalg.inv(lat)).dot(connections)
        assert is_pos_def(theta_obs)
        thetas_obs.append(theta_obs)

        sigma = np.linalg.inv(theta_obs)
        assert is_pos_def(sigma)
        sigmas.append(sigma)
        sampless.append(random_state.multivariate_normal(np.zeros(p), sigma, size=n))

    return (locations, thetas_O, thetas_H, thetas_obs, connections, sigmas, sampless)


def make_ell(n_dim_obs=100, n_dim_lat=10):
    """Make ell matrix."""
    K_HO = np.zeros((n_dim_lat, n_dim_obs))
    for i in range(n_dim_lat):
        percentage = int(n_dim_obs * 0.99)
        indices = np.random.randint(0, high=n_dim_obs, size=percentage)
        K_HO[i, indices] = np.random.rand(percentage) * (0.98 / (n_dim_lat + n_dim_obs) / 2)

    K_HO /= np.sum(K_HO, axis=1)[:, None] / 2.0
    L = K_HO.T.dot(K_HO)
    # print("{}%".format(np.nonzero(L)[0].size / L.size))
    assert is_pos_semidef(L)
    assert np.linalg.matrix_rank(L) == n_dim_lat
    # from sklearn.datasets import make_low_rank_matrix
    # L = make_low_rank_matrix(n_dim_obs, n_dim_obs, effective_rank=n_dim_lat)
    # L = (L + L.T) / 2.
    # print L
    return L, K_HO


def make_starting(n_dim_obs=100, n_dim_lat=10, degree=2, normalize=False, eps=1e-2):
    """Generate starting theta, theta_observed, L, K_HO."""
    L, K_HO = make_ell(n_dim_obs, n_dim_lat)

    if normalize:
        theta = np.zeros((n_dim_obs, n_dim_obs))
        for i in range(n_dim_obs):
            possible_idx = list(
                set(range(n_dim_obs))
                - (set(np.nonzero(theta[i, :])[0]) | set(np.where(np.count_nonzero(theta, axis=1) > degree)[0]))
            )
            if not possible_idx:
                continue
            n_choice = degree - (np.count_nonzero(theta[i, :]) - 1)
            if n_choice > 0:
                indices = np.random.choice(possible_idx, n_choice)
            theta[i, indices] = theta[indices, i] = 1.0 / degree

        theta.flat[:: n_dim_obs + 1] = np.sum(theta, axis=1) + 0.002

    else:
        theta = np.eye(n_dim_obs)
        for i in range(n_dim_obs):
            possible_idx = list(
                set(range(n_dim_obs))
                - (set(np.nonzero(theta[i, :])[0]) | set(np.where(np.count_nonzero(theta, axis=1) > degree)[0]))
            )
            if not possible_idx:
                continue
            n_choice = degree - (np.count_nonzero(theta[i, :]) - 1)
            if n_choice > 0:
                indices = np.random.choice(possible_idx, n_choice)
                theta[i, indices] = theta[indices, i] = 0.5 / degree

    if not is_pos_def(theta):
        theta += np.diag(np.sum(theta, axis=1) + eps)
    if not is_pos_def(theta - L):
        theta += np.diag(eps + np.diag(L))
    assert is_pos_def(theta - L)
    return theta, theta - L, L, K_HO


def _update_theta_l2(theta_old, n_dim_obs, degree, epsilon, keep_sparsity=False, indices=None):
    addition = np.zeros_like(theta_old)
    for i in range(n_dim_obs):
        if keep_sparsity:
            ii = indices[i]
        else:
            ii = np.random.randint(0, n_dim_obs, size=degree)
        addition[i, ii] = np.random.randn(len(ii))
    addition[np.triu_indices(n_dim_obs)[::-1]] = addition[np.triu_indices(n_dim_obs)]
    addition *= epsilon / np.linalg.norm(addition)
    np.fill_diagonal(addition, 0)
    theta = theta_old + addition
    theta[np.abs(theta) < 2 * epsilon / n_dim_obs] = 0
    return theta


def _update_theta_l1(theta_init, no, n_dim_obs, eps=1e-2):
    theta = theta_init.copy()
    rows = np.zeros(no)
    cols = np.zeros(no)
    while np.any(rows == cols):
        rows = np.random.randint(0, n_dim_obs, no)
        cols = np.random.randint(0, n_dim_obs, no)
    for r, c in zip(rows, cols):
        theta[r, c] = np.random.choice([0.12, 0, 0]) if theta[r, c] == 0 else 0.06  # np.random.rand(1) * .35
        theta[c, r] = theta[r, c]
    if not is_pos_def(theta):
        theta += np.diag(np.sum(theta, axis=1) + eps)
    return theta


def _update_ell_l2(K_HO_old, epsilon, n_dim_obs):
    K_HO = K_HO_old.copy()
    addition = np.random.rand(*K_HO.shape)
    addition *= epsilon / np.linalg.norm(addition)
    K_HO += addition
    K_HO /= np.sum(K_HO, axis=1)[:, None] / 2.0
    # K_HO *= 0.12
    K_HO[np.abs(K_HO) < epsilon / n_dim_obs] = 0
    return K_HO.T.dot(K_HO), K_HO


def make_covariance(
    n_dim_obs=100,
    n_dim_lat=10,
    T=10,
    update_ell="l1",
    update_theta="l1",
    normalize_starting_matrices=True,
    degree=2,
    epsilon=1e-2,
    keep_sparsity=False,
    proportional=False,
):
    """Generate a list of T covariances and sparse precisions."""
    no = int(np.ceil(n_dim_obs / 20)) if proportional else 1

    theta, theta_observed, L, K_HO = make_starting(n_dim_obs, n_dim_lat, degree, normalize=normalize_starting_matrices)

    thetas = [theta]
    thetas_obs = [theta_observed]
    ells = [L]
    K_HOs = [K_HO]

    idx = [np.nonzero(row)[0] for row in theta] if keep_sparsity else None
    for i in range(1, T):
        if update_theta == "l1":
            if keep_sparsity:
                warnings.warn("keep_sparsity is specified but is not " "implemented with l1.")
            theta = _update_theta_l1(thetas[-1], no, n_dim_obs)
        elif update_theta == "l2":
            theta = _update_theta_l2(thetas[-1], n_dim_obs, degree, epsilon, keep_sparsity=keep_sparsity, indices=idx)
        else:
            raise ValueError(update_theta)

        if update_ell == "fixed" or n_dim_lat < 1:
            pass  # L is fixed or empty
        elif update_ell == "l1":
            K_HO = K_HOs[-1].copy()
            picks = np.random.permutation(K_HO.size)[:no]
            K_HO = K_HO.ravel()
            for p in picks:
                K_HO[p] = np.random.choice([0.12, 0, 0]) if K_HO[p] == 0 else 0
            K_HO = np.reshape(K_HO, (n_dim_lat, n_dim_obs))
            L = K_HO.T.dot(K_HO)
        elif update_ell == "l2":
            L, K_HO = _update_ell_l2(K_HOs[-1], epsilon, n_dim_obs)
        elif update_ell == "yuan":
            K_HO = K_HOs[-1].copy()
            c = np.random.randint(0, n_dim_obs, 1)
            r = np.random.randint(0, n_dim_lat, 1)
            K_HO[r, c] = 0.12 if K_HO[r, c] == 0 else 0
            L = K_HO.T.dot(K_HO)
        else:
            raise ValueError(update_ell)

        assert np.linalg.matrix_rank(L) == n_dim_lat
        assert is_pos_semidef(L)
        if not is_pos_def(theta - L):
            theta += np.diag(epsilon + np.diag(L))

        thetas.append(theta)
        thetas_obs.append(theta - L)
        ells.append(L)
        K_HOs.append(K_HO)

    return thetas, thetas_obs, ells


def make_fixed_sparsity(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Generate precisions with a fixed L matrix and sparsity."""
    degree = kwargs.get("degree", 2)
    epsilon = kwargs.get("epsilon", 1e-2)
    t = kwargs.get("changing_time", T / 2.0)
    theta, theta_observed, L, K_HO = make_starting(n_dim_obs, n_dim_lat, degree, normalize=False)

    theta = np.abs(theta)
    theta_observed = theta - L
    nonzeros = np.nonzero(theta - np.diag(theta))

    thetas = [theta]
    thetas_obs = [theta_observed]

    for i in range(1, T):
        theta = thetas[-1].copy()
        if t < T:
            theta[nonzeros] += np.random.randn(nonzeros[0].size) * 0.1
        else:
            theta[nonzeros] -= np.random.randn(nonzeros[0].size) * 0.1
        theta = np.abs(theta)
        theta = (theta + theta.T) / 2.0
        theta[theta < epsilon] = 0
        theta.flat[:: n_dim_obs + 1] = np.sum(np.abs(theta), axis=1) + np.sum(np.abs(L), axis=1) + 0.1
        nonzeros = np.nonzero(theta - np.diag(theta))

        theta_observed = theta - L
        assert is_pos_def(theta_observed)
        thetas.append(theta)
        thetas_obs.append(theta_observed)

    return thetas, thetas_obs, np.array([L] * T)


def make_sin(n_dim_obs, n_dim_lat, T, shape="smooth", closeness=1, normalize=False, **kwargs):
    """Generate list of sparse precision matrices that change periodically."""
    upper_idx = np.triu_indices(n_dim_obs, 1)
    n_interactions = len(upper_idx[0])
    x = np.tile(np.linspace(0, (T - 1.0) / closeness, T), (n_interactions, 1))
    phase = np.random.rand(n_interactions, 1)
    freq = np.random.rand(n_interactions, 1) - 0.50
    A = (np.random.rand(n_interactions, 1) + 1) / 2.0

    if shape == "smooth":
        y = A * np.sin(2.0 * np.pi * freq * x + phase)
    else:
        y = A * signal.square(2 * np.pi * freq * x + phase, duty=0.5)

    # threshold
    y = np.maximum(y, 0)

    Y = np.array([squareform(y[:, j]) + np.diag(np.sum(squareform(y[:, j]), axis=1)) for j in range(y.shape[1])])

    if normalize:
        map(normalize_matrix, Y)  # in place
    # assert positive_definite(Y)
    ensure_posdef(Y)

    return Y, Y, np.zeros_like(Y)


def make_sin_cos(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Variables follow sin and cos evolution. L is fixed."""
    degree = kwargs.get("degree", 2)
    eps = kwargs.get("epsilon", 1e-2)
    L, K_HO = make_ell(n_dim_obs, n_dim_lat)

    phase = np.random.randn(n_dim_obs, n_dim_obs) * np.pi
    upper_idx_diag = np.triu_indices(n_dim_obs)
    phase[upper_idx_diag[::-1]] = phase[upper_idx_diag]

    upper_idx = np.triu_indices(n_dim_obs, 1)
    clip = np.zeros((n_dim_obs, n_dim_obs))
    picks = np.random.permutation(len(upper_idx[0]))
    dim = int(len(upper_idx[0]) * degree)
    picks = picks[:dim]
    clip1 = clip[upper_idx].ravel()
    clip1[picks] = 1
    clip[upper_idx[::-1]] = clip[upper_idx] = clip1

    thetas = np.array([np.eye(n_dim_obs) for i in range(T)])

    x = np.linspace(0, 2 * np.pi, T)
    for i in range(T):
        for r in range(n_dim_obs):
            for c in range(n_dim_obs):
                if r == c:
                    continue
                if clip[r, c]:
                    thetas[i, r, c] = np.sin((x[i] + phase[r, c]) / T ** 2)
                else:
                    thetas[i, r, c] = np.sin((x[i] + phase[r, c]))
        thetas[i][clip == 1] = np.clip(thetas[i][clip == 1], 0, 1)
        thetas[i][np.abs(thetas[i]) < eps] = 0

        assert is_pos_def(thetas[i])
        theta_observed = thetas[i] - L
        assert is_pos_def(theta_observed)
        thetas_obs = [theta_observed]

    return thetas, thetas_obs, np.array([L] * T)


def make_fede(n_dim_obs=3, n_dim_lat=2, T=10, epsilon=1e-3, n_samples=50, **kwargs):
    """Generate dataset (new version)."""
    b = np.random.rand(1, n_dim_obs)
    es, Q = np.linalg.eigh(b.T.dot(b))  # Q random

    b = np.random.rand(1, n_dim_obs)
    es, R = np.linalg.eigh(b.T.dot(b))  # R random

    start_sigma = np.random.rand(n_dim_obs) + 1
    start_lamda = np.zeros(n_dim_obs)
    idx = np.random.randint(n_dim_obs, size=n_dim_lat)
    start_lamda[idx] = np.random.rand(n_dim_lat)

    Ks = []
    Ls = []
    Kobs = []

    for i in range(T):
        K = np.linalg.multi_dot((Q, np.diag(start_sigma), Q.T))
        L = np.linalg.multi_dot((R, np.diag(start_lamda), R.T))

        K[np.abs(K) < epsilon] = 0  # enforce sparsity on K

        # assert is_pos_def(K - L)
        # assert is_pos_semidef(L)

        start_sigma += 1 + np.random.rand(n_dim_obs)
        start_lamda[idx] += np.random.rand(n_dim_lat) * 2 - 1
        start_lamda = np.maximum(start_lamda, 0)

        Ks.append(K)
        Ls.append(L)
        Kobs.append(K - L)

    return Ks, Kobs, Ls


def make_sparse_low_rank(n_dim_obs=3, n_dim_lat=2, T=10, epsilon=1e-3, **kwargs):
    """Generate dataset (new new version)."""
    from sklearn.datasets import make_sparse_spd_matrix, make_low_rank_matrix

    K = make_sparse_spd_matrix(n_dim_obs)
    L = make_low_rank_matrix(n_dim_obs, n_dim_obs, effective_rank=n_dim_lat)

    Ks = [K]
    Ls = [L]
    Kobs = [K - L]

    for i in range(1, T):
        K = K + make_sparse_spd_matrix(n_dim_obs)
        L = L + make_low_rank_matrix(n_dim_obs, n_dim_obs, effective_rank=n_dim_lat)

        # assert is_pos_def(K - L)
        # assert is_pos_semidef(L)

        Ks.append(K)
        Ls.append(L)
        Kobs.append(K - L)

    return Ks, Kobs, Ls


def make_ma_xue_zou(n_dim_obs=12, n_latent=3, T=1, epsilon=1e-3, sparsity=0.1, **kwargs):
    """Generate the dataset as in Ma, Xue, Zou (2012)."""
    # p = n_dim_obs + n_latent  # int(n_dim_obs * 0.05)
    p = n_dim_obs + int(n_dim_obs * 0.05)
    po = n_dim_obs
    ph = p - n_dim_obs
    W = np.zeros((p, p))
    non_zeros = int(round(p * p * sparsity))
    picks = np.random.permutation(p * p)[:non_zeros]
    W = W.ravel(order="F")
    W[picks] = np.random.randn(non_zeros)
    W = np.reshape(W, (p, p), order="F")

    C = W.T.dot(W)
    C[:po, po:] += 0.5 * np.random.randn(po, ph)
    C = (C + C.T) / 2.0

    C = np.clip(C - np.diag(np.diag(C)), -1, 1)
    eig, Q = np.linalg.eigh(C)
    K = C + max(-1.2 * np.min(eig), 0.001) * np.eye(p)
    K_O = K[:po, :po]
    K_OH = K[:po, po:]
    K_HO = K[po:, :po]
    K_H = K[po:, po:]

    # L = np.divide(K_OH, K_H.dot(K_HO))
    assert np.allclose(K_OH, K_HO.T)
    L = np.linalg.multi_dot((K_OH, np.linalg.inv(K_H), K_HO))
    K_O_tilde = K_O - L
    assert is_pos_def(K_O_tilde)
    assert is_pos_semidef(K_H)
    assert np.linalg.matrix_rank(L) == ph
    # print(ph)

    N = 5 * po
    print("Note that, with this method, the n_samples should be %d" % N)
    return [K_O] * T, [K_O_tilde] * T, [L] * T


def make_ma_xue_zou_rand_k(n_dim_obs=12, n_latent=3, T=1, epsilon=1e-3, sparsity=0.1, **kwargs):
    """Generate the dataset as in Ma, Xue, Zou (2012)."""
    # p = n_dim_obs + n_latent  # int(n_dim_obs * 0.05)
    p = n_dim_obs + int(n_dim_obs * 0.05)
    po = n_dim_obs
    nnzr = int(sparsity * (np.triu_indices(p, 1)[0].size))

    # Generate A, the original inverse covariance
    A = np.eye(p)
    idx = np.vstack(np.triu_indices(p, 1))
    idx = idx[:, np.random.choice(idx.shape[1], nnzr, replace=False)]
    idx = (idx[0], idx[1])
    A[idx] = np.sign(np.random.rand(nnzr) - 0.5)
    A[np.triu_indices(p, 1)[::-1]] = A[np.triu_indices(p, 1)]

    # A is the gound truth inverse covariance matrix
    K = A.dot(A.T) + 1e-6 * np.eye(p)
    K = A
    K_O = K[:po, :po]
    K_OH = K[:po, po:]
    K_HO = K[po:, :po]
    K_H = K[po:, po:]
    L = np.linalg.multi_dot((K_OH, np.linalg.inv(K_H), K_HO))
    K_O_tilde = K_O - L

    N = 5 * po
    print("Note that, with this method, the n_samples should be %d" % N)
    return [K_O] * T, [K_O_tilde] * T, [L] * T
