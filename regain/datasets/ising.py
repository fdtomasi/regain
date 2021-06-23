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
# implementation taken from the R library rIsing
import sys

import itertools

import numpy as np
import warnings


def update_ising_l1(theta_init, no, n_dim_obs, responses=[-1, 1]):
    """Generates temporal matrices using l1 update."""
    theta = theta_init.copy()
    rows = np.zeros(no)
    cols = np.zeros(no)
    while np.any(rows == cols):
        rows = np.random.randint(0, n_dim_obs, no)
        cols = np.random.randint(0, n_dim_obs, no)
        for r, c in zip(rows, cols):
            theta[r, c] = np.random.choice(responses) if theta[r, c] == 0 else np.random.choice([theta[r, c], 0])
            theta[c, r] = theta[r, c]
    assert np.all(theta == theta.T)
    return theta


def ising_theta_generator(
    n_dim_obs=10,
    n=100,
    T=10,
    mode="l1",
    time_on_axis="first",
    change=1,
    responses=[-1, 1],
    random_graph="scale-free",
    probability=0.3,
    degree=2,
):
    """Generates adjacency matrix for Ising graphical model.

    Parameters
    ----------
    n_dim_obs: int optional default 10
        Number of variables.

    n: int, optional default=100
        Number of samples.

    T: int, optional default=10
        Number of times.

    mode: string optional default 'l1'
        Type of temporal updates. For now no other choices are provided.

    change: int, optional default 1
        How many edges to change at each time.

    responses: list, optional default [-1,1]
        Possibilities are [-1, 1] or [0,1]

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
        graph = nx.adjacency_matrix(graph).todense()
    elif random_graph.lower() == "scale-free":
        graph = nx.random_graphs.barabasi_albert_graph(n=n_dim_obs, m=degree)
        graph = nx.adjacency_matrix(graph).todense()
    elif random_graph.lower() == "small-world":
        graph = nx.random_graphs.watts_strogatz_graph(n=n_dim_obs, k=degree, p=probability)
        graph = nx.adjacency_matrix(graph).todense()
    else:
        graph = np.random.choice(np.array([-0.5, 0, 0.5]), size=(n_dim_obs, n_dim_obs), replace=True)
        graph += graph.T
        np.fill_diagonal(graph, 0)
        graph[np.where(graph > 0)] = 1
        graph[np.where(graph < 0)] = -1
    thetas = [graph]

    for t in range(1, T):
        if mode == "switch":
            raise ValueError("Still not implemented")
        elif mode == "diffusion":
            raise ValueError("Still not implemented")
        elif mode == "l1":
            theta_t = update_ising_l1(thetas[-1], change, thetas[-1].shape[0])
        else:
            warnings.warn("Mode not implemented. Using l1.")
            theta_t = update_ising_l1(thetas[-1], change, thetas[-1].shape[0])
        thetas.append(theta_t)
    return thetas


def hamiltonian(theta, state, thresholds):
    res = 0
    n = theta.shape[0]
    for i in range(n):
        res -= thresholds[i] * state[i]
        res += np.sum([-theta[i, j] * state[i] * state[j] for j in range(n) if j != i])

    return res


def direct_sampling(theta, thresholds, n=1000, beta=1, responses=[0, 1]):

    if not np.all(theta == theta.T):
        raise ValueError("The input graph must be symmetric")
    if len(responses) != 2:
        raise ValueError("The responses must be two numbers.")

    np.fill_diagonal(theta, 0)

    states = [list(seq) for seq in itertools.product(responses, repeat=10)]
    probabilities = np.exp(-beta * np.array([hamiltonian(theta, s, thresholds) for s in states]))
    probabilities /= np.sum(probabilities)
    ixs = np.random.choice(np.arange(0, len(states)), n, replace=False, p=probabilities)
    return [states[i] for i in ixs]


def p_plus_minmax(i, theta, state, thresholds, beta, responses):

    H0 = np.ones(2) * thresholds[i] * responses[0]
    H1 = np.ones(2) * thresholds[i] * responses[1]
    minimum = -sys.maxsize - 1

    n = theta.shape[0]
    two_opts = np.zeros(2)

    for j in range(n):
        if i == j:
            continue
        if state[j] != minimum:
            H0[0] += theta[i, j] * responses[0] * state[j]
            H0[1] += theta[i, j] * responses[0] * state[j]
            H1[0] += theta[i, j] * responses[1] * state[j]
            H1[1] += theta[i, j] * responses[1] * state[j]
        else:
            two_opts[0] = theta[i, j] * responses[1] * responses[0]
            two_opts[1] = theta[i, j] * responses[1] * responses[1]
            if two_opts[1] > two_opts[0]:
                H1[0] += two_opts[0]
                H1[1] += two_opts[1]

                H0[0] += theta[i, j] * responses[0] * responses[0]
                H0[1] += theta[i, j] * responses[0] * responses[1]
            else:
                H1[0] += two_opts[1]
                H1[1] += two_opts[0]
                H0[0] += theta[i, j] * responses[0] * responses[1]
                H0[1] += theta[i, j] * responses[0] * responses[0]

    res = np.zeros(2)
    res[0] = np.exp(beta * H1[0]) / (np.exp(beta * H0[0]) + np.exp(beta * H1[0]))
    res[1] = np.exp(beta * H1[1]) / (np.exp(beta * H0[1]) + np.exp(beta * H1[1]))

    return res


def ising_exact(
    graph, thresholds, n=1000, beta=1, max_iter=100, max_chain=100, responses=[0, 1], exact=False, constraints=None
):

    minimum = -sys.maxsize - 1
    p = graph.shape[0]
    state = np.ones(p) * minimum
    probabilities = []
    min_trials = 0

    while np.any(state == minimum):

        probabilities.append(np.random.uniform(low=0, high=1, size=(max_iter, p)))

        for i in range(p):
            state[i] = minimum if exact else responses[0] if np.random.uniform() > 0.5 else responses[1]

        for trial in range(min_trials, -1, -1):
            for iter_ in range(max_iter):
                current_prob = probabilities[trial]
                for i in range(p):
                    prob = current_prob[iter_, i]
                    P = p_plus_minmax(i, graph, state, thresholds, beta, responses)
                    if prob < P[0]:
                        state[i] = responses[1]
                    elif prob >= P[1]:
                        state[i] = responses[0]
                    else:
                        state[i] = minimum

        min_trials += 1
    return state


def p_plus(i, theta, state, thresholds, beta, responses):
    H0 = thresholds[i] * responses[0]
    H1 = thresholds[i] * responses[1]

    p = theta.shape[0]

    for j in range(p):
        if i == j:
            continue
        H0 += theta[i, j] * responses[0] * state[j]
        H1 += theta[i, j] * responses[1] * state[j]

    return np.exp(beta * H1) / (np.exp(beta * H0) + np.exp(beta * H1))


def ising_metropolis_hastings(graph, thresholds, beta, max_iter, responses, constraints):
    p = graph.shape[0]
    state = [responses[1] if np.random.uniform() < 0.5 else responses[0] for i in range(p)]

    minimum = -sys.maxsize - 1

    for i in range(p):
        if constraints[i] != minimum:
            state[i] = constraints[i]
    for iter_ in range(max_iter):
        for i in range(p):
            if constraints[i] != minimum:
                continue

            uniform = np.random.uniform()
            P = p_plus(i, graph, state, thresholds, beta, responses)
            if uniform < P:
                state[i] = responses[1]
            else:
                state[i] = responses[0]

    return np.array(state)


def ising_sampler(
    theta, thresholds, n=1000, beta=1, max_iter=100, responses=[0, 1], method="MH", CFTP_retry=10, constraints=None
):
    """Given an adjacency matrix samples form the related distribution.

    Parameters
    ----------
    theta: array-like, (n_dim_obs, n_dim_obs)
        Number of variables.

    n: int, optional default=100
        Number of samples.

    method: string optional default 'MH'
        Type of sampling. Options are 'MH', 'CFTP', 'direct'

    responses: list, optional default [-1,1]
        Possibilities are [-1, 1] or [0,1]


    Returns
    --------
    array-like:
        Matrix of shape (n, n_dim_obs)
    """

    assert np.all(theta == theta.T), "The input graph must be symmetric"
    if len(responses) != 2:
        raise ValueError("The responses must be two numbers.")
    np.fill_diagonal(theta, 0)
    minimum = -sys.maxsize - 1
    p = theta.shape[0]

    if constraints is None:
        constraints = np.ones((n, p)) * minimum

    # CFTP method
    if method.upper() == "CFTP":

        for iter_ in range(CFTP_retry - 1):
            res = np.zeros((n, p))
            for s in range(n):
                state = ising_exact(
                    theta,
                    thresholds,
                    n=n,
                    beta=beta,
                    max_iter=max_iter,
                    responses=responses,
                    exact=True,
                    constraints=constraints,
                )
                res[s, :] = state
            if not np.any(np.isnan(res)):
                break
        else:
            print(
                "NA's detected. No CFTP sample was drawn after 100 "
                "couplings. Use higher max_iter value or method='MH' for "
                "inexact sampling."
            )

    # Metropolis Hastings
    elif method.upper() == "MH":
        res = ising_metropolis_hastings(theta, thresholds, beta, max_iter, responses, constraints[0, :])
        for s in range(1, n):
            state = ising_metropolis_hastings(theta, thresholds, beta, max_iter, responses, constraints[s, :])
            res = np.vstack((res, state))

    # Direct sampling
    elif method.upper() == "DIRECT":
        res = direct_sampling(theta, thresholds, n=n, beta=beta, responses=responses)
    # Unknown method
    else:
        raise ValueError("The method could be either MH, CFTP or DIRECT")

    return res
