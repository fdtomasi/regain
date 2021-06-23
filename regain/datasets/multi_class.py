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

import networkx as nx
import numpy as np
from sklearn.utils import Bunch

from regain.datasets.ising import ising_sampler
from regain.datasets.poisson import poisson_sampler


def make_multiclass_dataset(
    n_samples=100,
    n_dim_obs=100,
    n_classes=5,
    _type="scale-free",
    random_state=None,
    n_edges=2,
    probability=0.2,
    distribution="gaussian",
):
    """Generate a synthetic dataset.

    Parameters
    ----------
    n_samples: int,
        number of samples to generate
    n_dim_obs: int,
        number of observed variables of the graph
    n_classes: int,
        number of classes
    _type: string, default='scale-free'
        Type of random graph used for the initial generation. Options are
        'scale-free', 'erdos-renyi'.
    random_state: default=None
    n_edges: int, default=2
        Parameter for the scale-free random model, number of edges to attach
        for every new node in the network.
    probability: float in interval [0,1], default=0.2
        Parameter for the Erdos-Renyi random model, probability of edge
        creation.
    distribution: string or list, default='gaussian'
        The distribution considered for the generation of data . If it is a
        string it can either be one among 'gaussian', 'ising', 'poisson'.
        If it is a list it can be a combination of the three.

    Returns
    -------
    dict:
        For each distribution in the parameter distribution returns a tuple
        (
        data: list of array of length n_classes and each array of dimension
                n_samples x n_dim_obs
        thetas: list of networks of length n_classes, each network an adjacency
                matrix of shape n_dim_obs x n_dim_obs,
        X: array-like, shape=(n_samples*n_classes, n_dim_obs) data stucked in
            a unique array,
        y: array-like, shape=(n_samples*n_classes,) the labels of each sample
            to which class it belongs.
        )
    list:
        List of binarised version of the random networks.
    """
    res = generate_multiple_class_dataset(
        n_dim_obs=n_dim_obs,
        n_edges=n_edges,
        probability=probability,
        n_classes=n_classes,
        _type=_type,
        distribution=distribution,
        random_state=random_state,
    )
    return sample(n_dim_obs=n_dim_obs, n_classes=n_classes, n_samples=n_samples, networks=res), res["binary"]


def generate_multiple_class_dataset(
    n_dim_obs=10,
    n_edges=2,
    probability=0.2,
    n_classes=5,
    _type="scale-free",
    distribution="gaussian",
    random_state=None,
):
    """Generate networks for a multi-class problem.

    Parameters
    ----------
    n_dim_obs: int,
        number of observed variables of the graph
    n_classes: int,
        number of classes
    _type: string, default='scale-free'
        Type of random graph used for the initial generation. Options are
        'scale-free', 'erdos-renyi'.
    random_state: default=None
    n_edges: int, default=2
        Parameter for the scale-free random model, number of edges to attach
        for every new node in the network.
    probability: float in interval [0,1], default=0.2
        Parameter for the Erdos-Renyi random model, probability of edge
        creation.
    distribution: string or list, default='gaussian'
        The distribution considered for the generation of data . If it is a
        string it can either be one among 'gaussian', 'ising', 'poisson'.
        If it is a list it can be a combination of the three.

    Returns
    -------
    dict:
        For each distribution in the parameter distribution returns a list of
        length n_classes, each network an adjacency matrix of
        shape=(n_dim_obs, n_dim_obs), plus the binarised version of the
        networks.
    """
    if _type == "scale-free":
        graph = nx.random_graphs.barabasi_albert_graph(n=n_dim_obs, m=n_edges, seed=random_state)
    else:
        graph = nx.random_graphs.erdos_renyi_graph(n=n_dim_obs, p=probability, seed=random_state)

    binaries = [nx.adjacency_matrix(graph).todense()]
    zeros = np.where(binaries[0] == 0)
    nonzero = np.where(binaries[0] != 0)
    to_add = int(0.1 * zeros[0].shape[0])
    to_remove = int(0.1 * nonzero[0].shape[0])
    for i in range(n_classes - 1):
        K_new = binaries[0].copy()
        _to_add = np.random.choice(np.arange(zeros[0].shape[0]), to_add, replace=False)
        _to_remove = np.random.choice(np.arange(nonzero[0].shape[0]), to_remove, replace=False)
        for ta in _to_add:
            a = np.random.choice([0, 1], p=[0.2, 0.8])
            K_new[zeros[0][i], zeros[1][i]] = a
            K_new[zeros[1][i], zeros[0][i]] = a
        for ta in _to_remove:
            a = np.random.choice([0, 1], p=[0.8, 0.2])
            K_new[nonzero[0][i], nonzero[1][i]] = a
            K_new[nonzero[1][i], nonzero[0][i]] = a
        binaries.append(K_new)

    res = {}
    if "gaussian" in distribution:
        Ks_gaussian = [b.copy().astype(np.float32) for b in binaries]
        A = np.random.rand(n_dim_obs, n_dim_obs) - 0.5
        A = (A + A.T) / 2
        for b, k in zip(binaries, Ks_gaussian):
            k[np.where(b != 0)] = A[np.where(b != 0)]
            np.fill_diagonal(k, 1)
        res["gaussian"] = Ks_gaussian

    if "poisson" in distribution:
        res["poisson"] = [b.copy().astype(np.float32) for b in binaries]

    if "ising" in distribution:
        Ks_ising = [b.copy().astype(np.float32) for b in binaries]
        A = np.random.rand(n_dim_obs, n_dim_obs) - 0.5
        A = np.sign((A + A.T))
        for b, k in zip(binaries, Ks_ising):
            k[np.where(b != 0)] = A[np.where(b != 0)]
            np.fill_diagonal(k, 1)
        res["ising"] = Ks_ising
    res["binary"] = binaries

    return res


def sample(n_dim_obs=10, n_classes=5, n_samples=100, networks=dict()):
    """Generate a synthetic dataset.

    Parameters
    ----------
    n_samples: int,
        number of samples to generate
    n_dim_obs: int,
        number of observed variables of the graph
    n_classes: int,
        number of classes
    networks: dict,
        For each distribution in the parameter distribution returns a list of
        length n_classes, each network an adjacency matrix of
        shape=(n_dim_obs, n_dim_obs).

    Returns
    -------
    dict:
        For each distribution in the parameter distribution returns a tuple
        (
        data: list of array of length n_classes and each array of dimension
                n_samples x n_dim_obs
        thetas: list of networks of length n_classes, each network an adjacency
                matrix of shape n_dim_obs x n_dim_obs,
        X: array-like, shape=(n_samples*n_classes, n_dim_obs) data stucked in
            a unique array,
        y: array-like, shape=(n_samples*n_classes,) the labels of each sample
            to which class it belongs.
        )
    """

    def _gaussian(networks):
        sigmas = list(map(np.linalg.inv, networks))
        data = np.array(
            [np.random.multivariate_normal(np.zeros(n_dim_obs), sigma, size=n_samples) for sigma in sigmas]
        )
        X = np.vstack(data)
        y = np.repeat(range(n_classes), n_samples).astype(int)
        return Bunch(data=data, thetas=np.array(networks), X=X, y=y)

    def _ising(networks):
        data = np.array([ising_sampler(t, np.zeros(n_dim_obs), n=n_samples, responses=[-1, 1]) for t in networks])
        X = np.vstack(data)
        y = np.repeat(range(n_classes), n_samples).astype(int)

        return Bunch(data=data, thetas=np.array(networks), X=X, y=y)

    def _poisson(networks):
        data = np.array([poisson_sampler(t, variances=np.zeros(n_dim_obs), n_samples=n_samples) for t in networks])
        X = np.vstack(data)
        y = np.repeat(range(n_classes), n_samples).astype(int)

        return Bunch(data=data, thetas=np.array(networks), X=X, y=y)

    generation = {"gaussian": lambda x: _gaussian(x), "ising": lambda x: _ising(x), "poisson": lambda x: _poisson(x)}

    res = {}
    for k in networks.keys():
        if k == "binary":
            continue
        res[k] = generation[k](networks[k])

    return res
