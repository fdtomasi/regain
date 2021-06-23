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
"""Generate data for kernel-based classes as `KernelTimeGraphicalLasso`."""

from itertools import chain
from itertools import combinations

import numpy as np
from scipy import linalg
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import Bunch

from regain.datasets.gaussian import make_ell
from regain.norm import l1_od_norm
from regain.utils import is_pos_def


def make_exp_sine_squared(n_dim_obs=5, n_dim_lat=0, T=1, **kwargs):
    """Make precision matrices using a temporal sine kernel."""
    from regain.bayesian.gaussian_process_ import sample as samplegp
    from sklearn.gaussian_process import kernels

    L, K_HO = make_ell(n_dim_obs, n_dim_lat)

    periodicity = kwargs.get("periodicity", np.pi)
    length_scale = kwargs.get("length_scale", 2)
    epsilon = kwargs.get("epsilon", 0.5)
    sparse = kwargs.get("sparse", True)
    temporal_kernel = kernels.ExpSineSquared(periodicity=periodicity, length_scale=length_scale)(np.arange(T)[:, None])

    u = samplegp(temporal_kernel, p=n_dim_obs * (n_dim_obs - 1) // 2)[0]
    K, K_obs = [], []
    for uu in u.T:
        theta = squareform(uu)

        if sparse:
            # sparsify
            theta[np.abs(theta) < epsilon] = 0

        theta += np.diag(np.sum(np.abs(theta), axis=1) + 0.01)
        K.append(theta)

        assert is_pos_def(theta)
        theta_observed = theta - L
        assert is_pos_def(theta_observed)
        K_obs.append(theta_observed)

    thetas = np.array(K)
    return thetas, np.array(K_obs), np.array([L] * T)


def make_RBF(n_dim_obs=5, n_dim_lat=0, T=1, **kwargs):
    """Make precision matrices using a temporal RBF kernel."""
    from regain.bayesian.gaussian_process_ import sample as samplegp
    from sklearn.gaussian_process import kernels

    length_scale = kwargs.get("length_scale", 1.0)
    length_scale_bounds = kwargs.get("length_scale_bounds", (1e-05, 100000.0))
    epsilon = kwargs.get("epsilon", 0.8)
    sparse = kwargs.get("sparse", True)
    temporal_kernel = kernels.RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)(
        np.arange(T)[:, None]
    )

    n = n_dim_obs + n_dim_lat
    u = samplegp(temporal_kernel, p=n * (n - 1) // 2)[0]
    K = []
    for i, uu in enumerate(u.T):
        theta = squareform(uu)
        if sparse:
            theta_obs = theta[n_dim_lat:, n_dim_lat:]
            theta_lat = theta[:n_dim_lat, :n_dim_lat]
            theta_OH = theta[n_dim_lat:, :n_dim_lat]

            # sparsify
            theta_obs[np.abs(theta_obs) < epsilon] = 0
            theta_lat[np.abs(theta_lat) < epsilon / 3] = 0
            theta_OH[np.abs(theta_OH) < epsilon / 3] = 0
            theta[n_dim_lat:, n_dim_lat:] = theta_obs
            theta[:n_dim_lat, :n_dim_lat] = theta_lat
            theta[n_dim_lat:, :n_dim_lat] = theta_OH
            theta[:n_dim_lat, n_dim_lat:] = theta_OH.T
        if i == 0:
            inter_links = theta[n_dim_lat:, :n_dim_lat]
        theta[n_dim_lat:, :n_dim_lat] = inter_links
        theta[:n_dim_lat, n_dim_lat:] = inter_links.T
        theta += np.diag(np.sum(np.abs(theta), axis=1) + 0.01)
        K.append(theta)

        assert is_pos_def(theta)

    thetas = np.array(K)

    theta_obs = []
    ells = []
    for t in thetas:
        L = (
            theta[n_dim_lat:, :n_dim_lat]
            .dot(linalg.pinv(t[:n_dim_lat, :n_dim_lat]))
            .dot(theta[:n_dim_lat, n_dim_lat:])
        )
        theta_obs.append(t[n_dim_lat:, n_dim_lat:] - L)
        ells.append(L)
    return thetas, theta_obs, ells


def genInvCov(size, low=0.3, upper=0.6, portion=0.2, symmetric=True):
    """Generate sparse inverse covariance matrix."""
    import networkx as nx

    portion = portion / 2
    S = np.zeros((size, size))
    n_edges = int((size * (size - 1)) * portion)
    G = nx.gnm_random_graph(size, n_edges)
    for src, dest in G.edges:
        S[src, dest] = (np.random.randint(2) - 0.5) * 2 * (low + (upper - low) * np.random.rand(1)[0])
    if symmetric:
        S += S.T
    # vals = alg.eigvalsh(S)
    # S = S + (0.1 - vals[0])*np.identity(size)
    return S


def genRandInv(size, low=0.3, upper=0.6, portion=0.2):
    """Generate inverse covariance matrix."""
    S = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if np.random.rand() < portion:
                value = (np.random.randint(2) - 0.5) * 2 * (low + (upper - low) * np.random.rand(1)[0])
                S[i, j] = value
    return S


def make_ticc(num_blocks=5, n_dim_obs=5, n_dim_lat=0, sparsity_inv_matrix=0.5, rand_seed=None, **kwargs):
    if n_dim_lat != 0:
        raise ValueError("not supported")

    np.random.seed(rand_seed)
    size_blocks = n_dim_obs
    block_matrices = {}

    # Generate all the blocks
    for block in range(num_blocks):
        if block == 0:
            block_matrices[block] = genInvCov(
                size=size_blocks,
                portion=sparsity_inv_matrix,
                symmetric=(block == 0),
            )
        else:
            block_matrices[block] = genRandInv(size=size_blocks, portion=sparsity_inv_matrix)

    # Initialize the inverse matrix
    inv_matrix = np.zeros([num_blocks * size_blocks, num_blocks * size_blocks])

    # go through all the blocks
    for block_i in range(num_blocks):
        for block_j in range(num_blocks):
            block_num = np.abs(block_i - block_j)
            inv_matrix[
                block_i * size_blocks : (block_i + 1) * size_blocks,
                block_j * size_blocks : (block_j + 1) * size_blocks,
            ] = (
                block_matrices[block_num] if block_i > block_j else np.transpose(block_matrices[block_num])
            )

    # print out all the eigenvalues
    eigs, _ = np.linalg.eig(inv_matrix)
    lambda_min = min(eigs)

    # Make the matrix positive definite
    inv_matrix += (0.1 + abs(lambda_min)) * np.eye(size_blocks * num_blocks)

    eigs, _ = np.linalg.eig(inv_matrix)
    lambda_min = min(eigs)
    # print("Modified Eigenvalues are:", np.sort(eigs))

    return inv_matrix


def _block_matrix(matrix, r, c):
    S11 = matrix[:r, :c]
    S22 = matrix[r:, c:]
    S21 = matrix[r:, :c]
    S12 = S21.T
    return S11, S22, S21, S12


def make_ticc_dataset(
    clusters=(0, 1, 0),
    n_dim=3,
    w_size=5,
    break_points=None,
    n_samples=200,
    n_dim_lat=0,
    sparsity_inv_matrix=0.5,
    T=9,
    rand_seed=None,
    **kwargs
):
    """Generate data as the TICC method.

    Library implementation of `generate_synthetic_data.py`, original can be
    found at https://github.com/davidhallac/TICC
    """
    if (len(clusters) * n_samples) % T != 0:
        raise ValueError(
            "n_clusters * n_samples should be a multiple of n_times "
            "to avoid having samples in the same time period in different "
            "clusters"
        )

    id_cluster = np.repeat(np.asarray(list(clusters)), n_samples)
    y = np.repeat(np.arange(T), len(clusters) * n_samples // T)

    cluster_mean = np.zeros(n_dim)
    cluster_mean_stack = np.zeros(n_dim * w_size)

    clusters = np.unique(list(clusters))
    # Generate two inverse matrices
    precisions = {}
    covs = {}
    for i, cluster in enumerate(clusters):
        precisions[cluster] = make_ticc(
            rand_seed=i,
            num_blocks=w_size,
            n_dim_obs=n_dim,
            n_dim_lat=n_dim_lat,
            sparsity_inv_matrix=sparsity_inv_matrix,
            **kwargs
        )
        covs[cluster] = linalg.pinvh(precisions[cluster])

    # Data matrix
    X = np.empty((id_cluster.size, n_dim))
    precs = []
    n = n_dim
    for i, label in enumerate(id_cluster):
        # for num in range(old_break_pt, break_pt):
        if i == 0:
            # conditional covariance and mean
            cov_tom = covs[label][:n_dim, :n_dim]
            mean = cluster_mean_stack[n_dim * (w_size - 1) :]

        elif i < w_size:
            cov = covs[label][: (i + 1) * n, : (i + 1) * n]
            Sig11, Sig22, Sig21, Sig12 = _block_matrix(cov, i * n, i * n)
            Sig21Theta11 = Sig21.dot(linalg.pinvh(Sig11))
            cov_tom = Sig22 - Sig21Theta11.dot(Sig12)  # sigma2|1

            mean = cluster_mean + Sig21Theta11.dot(X[:i].flatten() - cluster_mean_stack[: i * n_dim])

        else:
            cov = covs[label][: w_size * n, : w_size * n]
            Sig11, Sig22, Sig21, Sig12 = _block_matrix(cov, (w_size - 1) * n, (w_size - 1) * n)
            Sig21Theta11 = Sig21.dot(linalg.pinvh(Sig11))
            cov_tom = Sig22 - Sig21Theta11.dot(Sig12)  # sigma2|1

            mean = cluster_mean + Sig21Theta11.dot(
                X[i - w_size + 1 : i].flatten() - cluster_mean_stack[: (w_size - 1) * n_dim]
            )

        X[i] = np.random.multivariate_normal(mean, cov_tom)
        precs.append(linalg.pinvh(cov_tom))

    id_cluster_group = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        # check samples at same time belong to a single cluster
        assert np.unique(id_cluster[idx]).size == 1
        id_cluster_group.append(id_cluster[idx][0])

    data = Bunch(
        X=X,
        y=y,
        id_cluster=id_cluster,
        covs=covs,
        precs=precs,
        id_cluster_group=np.asarray(id_cluster_group),
    )
    return data


def make_ticc_dataset_new(
    clusters=(0, 1, 0),
    n_dim=3,
    w_size=5,
    break_points=None,
    n_samples=200,
    n_dim_lat=0,
    sparsity_inv_matrix=0.5,
    T=9,
    rand_seed=None,
    **kwargs
):
    """Generate data as the TICC method.

    Library implementation of `generate_synthetic_data.py`, original can be
    found at https://github.com/davidhallac/TICC
    """
    # if (len(clusters) * n_samples) % T != 0:
    #     raise ValueError(
    #         'n_clusters * n_samples should be a multiple of n_times '
    #         'to avoid having samples in the same time period in different '
    #         'clusters')

    id_cluster = np.repeat(np.asarray(list(clusters)), n_samples)
    y = np.repeat(np.arange(T), len(clusters) * n_samples // T)

    size_blocks = n_dim
    num_blocks = n_samples
    bmc = {}

    for rand_seed in clusters:
        np.random.seed(rand_seed)

        bmc[rand_seed] = {
            block: (
                genInvCov(
                    size=size_blocks,
                    portion=sparsity_inv_matrix,
                    symmetric=True,
                )
                if block == 0
                else (genRandInv(size=size_blocks, portion=sparsity_inv_matrix))
                if block < w_size + 1
                else np.zeros((n_dim, n_dim))
            )
            for block in range(num_blocks)
        }

    def get_out_diag(r):
        return np.block(
            [
                [
                    (bmc[r].get(i, np.zeros((n_dim, n_dim))).T if col > j else bmc[r].get(i, np.zeros((n_dim, n_dim))))
                    for col, i in enumerate(range(j, num_blocks + j))
                ]
                for j in range(1, num_blocks + 1)
            ][::-1]
        )

    def get_out_zero(r):
        return np.zeros_like(get_out_diag(r))

    def get_diag(r):
        return np.block(
            [
                [
                    (bmc[r][i] if col > j else bmc[r][i].T)
                    for col, i in enumerate(chain(range(j, 0, -1), range(num_blocks - j)))
                ]
                for j in range(num_blocks)
            ]
        )

    inv = np.block(
        [
            [np.zeros_like(get_out_diag(c))] * max(0, k - 1)
            + ([get_out_diag(clusters[k - 1]).T] if k > 0 else [])
            + [get_diag(c)]
            + ([get_out_diag(c)] if k < len(clusters) - 1 else [])
            + [np.zeros_like(get_out_diag(c))] * max(len(clusters) - 2 - k, 0)
            for k, c in enumerate(clusters)
        ]
    )
    # Make the matrix positive definite
    eigs, _ = np.linalg.eig(inv)
    lambda_min = np.min(eigs)
    inv += (0.1 + abs(lambda_min)) * np.eye(inv.shape[0])
    C = linalg.pinvh(inv)

    # create data
    cluster_mean = np.zeros(n_dim)
    cluster_mean_stack = np.zeros((n_samples * len(clusters), n_dim))
    X = np.empty((id_cluster.size, n_dim))
    precs, sparse_precs = [], []
    for i, label in enumerate(id_cluster):
        cov = C[
            max(i - w_size, 0) * n_dim : (i + 1) * n_dim,
            max(i - w_size, 0) * n_dim : (i + 1) * n_dim,
        ]

        # conditional covariance and mean
        Sig11, Sig22, Sig21, Sig12 = _block_matrix(cov, min(i, w_size) * n_dim, min(i, w_size) * n_dim)
        T11, T22, T21, T12 = _block_matrix(
            inv[
                max(i - w_size, 0) * n_dim : (i + 1) * n_dim,
                max(i - w_size, 0) * n_dim : (i + 1) * n_dim,
            ],
            min(i, w_size) * n_dim,
            min(i, w_size) * n_dim,
        )
        # print(Sig11.shape, Sig22.shape, Sig21.shape, Sig12.shape)

        # when i = 0, Sig11 has shape (0,0), causing an error in pinvh
        Sig21Theta11 = Sig21.dot(linalg.pinvh(Sig11) if Sig11.size > 0 else Sig11)
        cov_tom = Sig22 - Sig21Theta11.dot(Sig12)  # sigma2|1

        mean = cluster_mean + Sig21Theta11.dot(
            X[max(i - w_size, 0) : i].flatten() - cluster_mean_stack[max(i - w_size, 0) : i].flatten()
        )

        X[i] = np.random.multivariate_normal(mean, cov_tom)
        precs.append(linalg.pinvh(cov_tom))
        sparse_precs.append(T22)

    id_cluster_group = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        # check samples at same time belong to a single cluster
        assert np.unique(id_cluster[idx]).size == 1
        id_cluster_group.append(id_cluster[idx][0])

    data = Bunch(
        X=X,
        y=y,
        id_cluster=id_cluster,
        precs=precs,
        sparse_precs=sparse_precs,
        id_cluster_group=np.asarray(id_cluster_group),
        inv=inv,
    )
    return data


def make_ticc_dataset_v3(
    clusters=(0, 1, 0),
    n_dim=3,
    w_size=5,
    break_points=None,
    n_samples=200,
    n_dim_lat=0,
    sparsity_inv_matrix=0.5,
    T=9,
    rand_seed=None,
    **kwargs
):
    """Generate data as the TICC method.

    Library implementation of `generate_synthetic_data.py`, original can be
    found at https://github.com/davidhallac/TICC
    """
    # if (len(clusters) * n_samples) % T != 0:
    #     raise ValueError(
    #         'n_clusters * n_samples should be a multiple of n_times '
    #         'to avoid having samples in the same time period in different '
    #         'clusters')

    # id_cluster = np.repeat(np.asarray(list(clusters)), n_samples)
    # y = np.repeat(np.arange(T), len(clusters) * n_samples // T)

    size_blocks = n_dim
    num_blocks = 1
    bmc = {}

    for rand_seed in np.unique(clusters):
        np.random.seed(rand_seed)

        bmc[rand_seed] = {
            block: (
                genInvCov(
                    size=size_blocks,
                    portion=sparsity_inv_matrix,
                    symmetric=True,
                )
                if block == 0
                else (genRandInv(size=size_blocks, portion=sparsity_inv_matrix))
                if block < w_size + 1
                else np.zeros((n_dim, n_dim))
            )
            for block in range(num_blocks)
        }

    def get_out_diag(r):
        return np.block(
            [
                [
                    (bmc[r].get(i, np.zeros((n_dim, n_dim))).T if col > j else bmc[r].get(i, np.zeros((n_dim, n_dim))))
                    for col, i in enumerate(range(j, num_blocks + j))
                ]
                for j in range(1, num_blocks + 1)
            ][::-1]
        )

    def get_out_zero(r):
        return np.zeros_like(get_out_diag(r))

    def get_diag(r):
        return np.block(
            [
                [
                    (bmc[r][i] if col > j else bmc[r][i].T)
                    for col, i in enumerate(chain(range(j, 0, -1), range(num_blocks - j)))
                ]
                for j in range(num_blocks)
            ]
        )

    inv = np.block(
        [
            [np.zeros_like(get_out_diag(c))] * max(0, k - 1)
            + ([get_out_diag(clusters[k - 1]).T] if k > 0 else [])
            + [get_diag(c)]
            + ([get_out_diag(c)] if k < len(clusters) - 1 else [])
            + [np.zeros_like(get_out_diag(c))] * max(len(clusters) - 2 - k, 0)
            for k, c in enumerate(clusters)
        ]
    )
    # Make the matrix positive definite
    eigs, _ = np.linalg.eig(inv)
    lambda_min = np.min(eigs)
    inv += (0.1 + abs(lambda_min)) * np.eye(inv.shape[0])
    C = linalg.pinvh(inv)

    # create data
    cluster_mean = np.zeros(n_dim)
    cluster_mean_stack = np.zeros((n_samples * len(clusters), n_dim))
    X = np.empty((n_samples * len(clusters), n_dim))
    precs, sparse_precs = [], []
    for i, _ in enumerate(clusters):
        cov = C[
            max(i - w_size, 0) * n_dim : (i + 1) * n_dim,
            max(i - w_size, 0) * n_dim : (i + 1) * n_dim,
        ]

        # conditional covariance and mean
        Sig11, Sig22, Sig21, Sig12 = _block_matrix(cov, min(i, w_size) * n_dim, min(i, w_size) * n_dim)
        _, T22, _, _ = _block_matrix(
            inv[
                max(i - w_size, 0) * n_dim : (i + 1) * n_dim,
                max(i - w_size, 0) * n_dim : (i + 1) * n_dim,
            ],
            min(i, w_size) * n_dim,
            min(i, w_size) * n_dim,
        )
        # print(Sig11.shape, Sig22.shape, Sig21.shape, Sig12.shape)

        # when i = 0, Sig11 has shape (0,0), causing an error in pinvh
        Sig21Theta11 = Sig21.dot(linalg.pinvh(Sig11) if Sig11.size > 0 else Sig11)
        cov_tom = Sig22 - Sig21Theta11.dot(Sig12)  # sigma2|1

        mean = cluster_mean + Sig21Theta11.dot(
            X[max(i - w_size, 0) : i].flatten() - cluster_mean_stack[max(i - w_size, 0) : i].flatten()
        )

        X[i * n_samples : (i + 1) * n_samples] = np.random.multivariate_normal(mean, cov_tom, size=n_samples)
        precs.append(linalg.pinvh(cov_tom))
        sparse_precs.append(T22)

    y = np.repeat(np.arange(len(clusters)), n_samples)
    id_cluster = np.repeat(clusters, n_samples)

    data = Bunch(
        X=X,
        y=y,
        id_cluster=id_cluster,
        precs=np.array(precs),
        sparse_precs=np.array(sparse_precs),
        inv=inv,
    )
    return data


def make_cluster_representative(
    n_dim=10,
    degree=2,
    n_clusters=3,
    T=15,
    n_samples=100,
    repetitions=False,
    cluster_series=None,
    shuffle=False,
):
    """Based on the cluster representative, generate similar graphs."""
    import networkx as nx

    cluster_reps = []
    adjacencies = []
    if cluster_series is not None:
        n_clusters = np.unique(cluster_series).size

    for i in range(n_clusters):
        representative = nx.random_regular_graph(d=degree, n=n_dim)
        A = nx.adjacency_matrix(representative).todense().astype(float)
        A[np.where(A != 0)] = np.random.rand(np.where(A != 0)[0].size) * 0.45
        np.fill_diagonal(A, 1)
        adjacencies.append(A)
        cluster_reps.append(representative)

    pos = np.arange(0, T, T // (n_clusters + 1))
    pos = list(pos) + [T - 1]

    if cluster_series is None:
        cluster_series = np.tile(range(n_clusters), (len(pos) // n_clusters) + 1)[: len(pos)]
        if shuffle:
            np.random.shuffle(cluster_series)
        # print(pos)
        # print(cluster_series)
        # pos = np.arange(0, T, T // (clusters + 1))
        # pos = list(pos) + [T - 1]
    else:
        assert len(cluster_series) == len(pos)
    #     a = np.where(cluster_series[:-1] != cluster_series[1:])[0] + 1
    #     T = len(cluster_series) # overwrites T
    #     pos = np.concatenate(([0], a, [T-1]))

    thetas = []
    for i in range(len(pos) - 1):
        # last one is always a representative
        how_many = int(pos[i + 1]) - int(pos[i]) - 1
        new_list = [adjacencies[cluster_series[i]]]
        target = adjacencies[cluster_series[i + 1]]

        for i in range(how_many):
            new = new_list[-1].copy()
            diffs = (new != 0).astype(int) - (target != 0).astype(int)
            diff = np.where(diffs != 0)
            if diff == ():
                break
            if i == 0:
                edges_per_change = int((np.nonzero(diffs)[0].shape[0] / 2) // (how_many + 1))
                if edges_per_change == 0:
                    edges_per_change += 1
            ixs = np.arange(diff[0].shape[0])
            np.random.shuffle(ixs)

            xs = diff[0][ixs[:edges_per_change]]
            ys = diff[1][ixs[:edges_per_change]]
            for j in range(xs.shape[0]):
                if diffs[xs[j], ys[j]] == -1:
                    new[xs[j], ys[j]] = np.random.rand(1) * 0.2
                    new[ys[j], xs[j]] = new[xs[j], ys[j]]
                else:
                    new[xs[j], ys[j]] = 0
                    new[ys[j], xs[j]] = 0
            new_list.append(new)

        thetas += new_list
    thetas.append(target)
    covs = [linalg.pinvh(t) for t in thetas]
    X = np.vstack([np.random.multivariate_normal(np.zeros(n_dim), c, size=n_samples) for c in covs])
    y = np.repeat(np.arange(len(covs)), n_samples)

    distances = squareform([l1_od_norm(t1 - t2) for t1, t2 in combinations(thetas, 2)])
    distances /= np.max(distances)
    labels_pred = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="precomputed", linkage="complete"
    ).fit_predict(distances)

    id_cluster = np.repeat(labels_pred, n_samples)
    data = Bunch(
        X=X,
        y=y,
        id_cluster=id_cluster,
        precs=np.array(thetas),
        thetas=np.array(thetas),
        sparse_precs=np.array(thetas),
        cluster_reps=cluster_reps,
        cluster_series=cluster_series,
    )
    return data
