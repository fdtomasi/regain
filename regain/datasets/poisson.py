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
import warnings

import numpy as np


def update_l1(G, how_many, n_dim_obs):
    """Generates temporal matrices using l1 update."""

    G = G.copy()
    rows = np.zeros(how_many)
    cols = np.zeros(how_many)
    while np.any(rows == cols):
        rows = np.random.randint(0, n_dim_obs, how_many)
        cols = np.random.randint(0, n_dim_obs, how_many)
        for r, c in zip(rows, cols):
            G[r, c] = np.random.choice([0, G[r, c]])
            G[c, r] = G[r, c]
    assert np.all(G == G.T)
    return G


def poisson_theta_generator(
    n_dim_obs=10, T=10, mode="l1", random_graph="erdos-renyi", probability=0.2, degree=3, n_to_change=3, **kwargs
):
    """Generates adjacency matix for Ising graphical model.

    Parameters
    ----------
    n_dim_obs: int optional default 10
        Number of variables.

    T: int, optional default=10
        Number of times.

    mode: string optional default 'l1'
        Type of temporal updates. For now no other choices are provided.

    n_to_change: int, optional default 1
        How many edges to change at each time.

    random_graph: string, optional default 'scale-free'
        Initial adjacency matrix graph type. Options 'scale-free',
        'erdos-renyi'

    probability: float, optional default 0.3
        Paramter for Erdos-Renyi random graph.

    degree: int, optional default 2
        Parameter for the scale free random graph.

    Returns
    --------
    list:
        List of adjaceny matrix of length T.
    """
    import networkx as nx

    if random_graph.lower() == "erdos-renyi":
        graph = nx.random_graphs.fast_gnp_random_graph(n=n_dim_obs, p=probability)
    elif random_graph.lower() == "scale-free":
        graph = nx.random_graphs.barabasi_albert_graph(n=n_dim_obs, m=degree)
    elif random_graph.lower() == "small-world":
        graph = nx.random_graphs.watts_strogatz_graph(n=n_dim_obs, k=degree, p=probability)
    else:
        graph = nx.random_graphs.gnm_random_graph(n=n_dim_obs, m=degree)
    graph = nx.adjacency_matrix(graph).todense()
    np.fill_diagonal(graph, 0)
    graphs = [graph]
    for t in range(1, T):
        if mode == "reshuffle":
            raise ValueError("Still not implemented")
        elif mode == "l1":
            graph_t = update_l1(graphs[-1], n_to_change, graphs[0].shape[0])
        else:
            warnings.warn("Mode not implemented. Using l1.")
            graph_t = update_l1(graphs[-1], n_to_change, graphs[0].shape[0])
        np.fill_diagonal(graph_t, 0)
        graphs.append(graph_t)

    return graphs


def _adjacency_to_A(graph, typ="full"):
    A = np.eye(graph.shape[0])
    for i in range(graph.shape[0] - 1):
        for j in range(i + 1, graph.shape[0]):
            if typ == "full" or graph[i, j] == 1:
                tmp = np.zeros((graph.shape[0], 1))
                tmp[np.array([i, j]), 0] = 1
                A = np.hstack((A, tmp))
    return A


def poisson_sampler(
    theta,
    variances=None,
    n_samples=100,
    _type="LPGM",
    random_graph="scale-free",
    _lambda=1,
    _lambda_noise=0.5,
    max_iter=200,
):
    """Given an adjacency matrix samples form the related distribution.

    Parameters
    ----------
    theta: array-like, (n_dim_obs, n_dim_obs)
        Number of variables.

    variances: list, optional default None
        List of length n_dim_obs with the variance of each variable.

    n_samples: int, optional default=100
        Number of samples.

    _type: string, optional default='LPGM'
        Type if local poisson or poisson.

    random_graph: string, optional default 'scale-free'
        Initial adjacency matrix graph type. Options 'scale-free',
        'erdos-renyi'

    _lambda: float, optional default 1
        Parameter of the conditional poisson distribution of each variable.

    _lambda_noise: float, optional default 0.5
        Parameter of the noise poisson distribution.

    max_iter: int, optional defatul 200
        Maximum iteration for Gibbs sampling procedure.

    Returns
    --------
    array-like:
        Matrix of shape (n, n_dim_obs)
    """
    n_dim_obs = theta.shape[0]
    if _type == "LPGM":
        A = _adjacency_to_A(theta, typ="scale-free")
        sigma = _lambda * theta
        ltri_sigma = sigma[np.tril_indices(sigma.shape[0], k=-1)]
        nonzero_sigma = ltri_sigma[np.where(ltri_sigma != 0)]
        aux = [_lambda] * theta.shape[0]
        Y_lambda = np.array(aux + nonzero_sigma.ravel().tolist()[0])

        Y = np.array([np.random.poisson(l, n_samples) for l in Y_lambda]).T
        X = Y.dot(A.T)

        # add noise
        X = X + np.random.poisson(_lambda_noise, size=(n_samples, n_dim_obs))

    else:  # Gibbs sampling
        if variances is None or variances.size != n_dim_obs:
            variances = np.zeros(n_dim_obs)
        variances.reshape(n_dim_obs, 1)

        X = np.random.poisson(1, size=(n_samples, n_dim_obs))

        for iter_ in range(max_iter):
            for i in range(n_dim_obs):
                selector = np.array([j for j in range(n_dim_obs) if j != i])
                par = np.exp(variances[i] + X[:, selector].dot(theta[selector, j]))
                X[:, i] = np.array([np.random.poisson(p) for p in par])

    return X
