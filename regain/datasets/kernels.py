from __future__ import division

import numpy as np
from scipy import linalg
from scipy.spatial.distance import squareform

from regain.utils import is_pos_def

from .gaussian import make_ell


def make_exp_sine_squared(n_dim_obs=5, n_dim_lat=0, T=1, **kwargs):
    from regain.bayesian.gaussian_process_ import sample as samplegp
    from scipy.spatial.distance import squareform
    from sklearn.gaussian_process import kernels

    L, K_HO = make_ell(n_dim_obs, n_dim_lat)

    periodicity = kwargs.get('periodicity', np.pi)
    length_scale = kwargs.get('length_scale', 2)
    epsilon = kwargs.get('epsilon', 0.5)
    sparse = kwargs.get('sparse', True)
    temporal_kernel = kernels.ExpSineSquared(
        periodicity=periodicity,
        length_scale=length_scale)(np.arange(T)[:, None])

    u = samplegp(temporal_kernel, p=n_dim_obs * (n_dim_obs - 1) // 2)[0]
    K, K_obs = [], []
    for uu in u.T:
        theta = squareform(uu)

        if sparse:
            # sparsify
            theta[np.abs(theta) < epsilon] = 0

        theta += np.diag(np.sum(np.abs(theta), axis=1) + 0.01)
        K.append(theta)

        assert (is_pos_def(theta))
        theta_observed = theta - L
        assert (is_pos_def(theta_observed))
        K_obs.append(theta_observed)

    thetas = np.array(K)
    return thetas, np.array(K_obs), np.array([L] * T)


def make_RBF(n_dim_obs=5, n_dim_lat=0, T=1, **kwargs):
    from regain.bayesian.gaussian_process_ import sample as samplegp
    from sklearn.gaussian_process import kernels

    length_scale = kwargs.get('length_scale', 1.0)
    length_scale_bounds = kwargs.get('length_scale_bounds', (1e-05, 100000.0))
    epsilon = kwargs.get('epsilon', 0.8)
    sparse = kwargs.get('sparse', True)
    temporal_kernel = kernels.RBF(
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds)(np.arange(T)[:, None])

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

        assert (is_pos_def(theta))

    thetas = np.array(K)

    theta_obs = []
    ells = []
    for t in thetas:
        L = theta[n_dim_lat:, :n_dim_lat].dot(
            linalg.pinv(t[:n_dim_lat, :n_dim_lat])).dot(
                theta[:n_dim_lat, n_dim_lat:])
        theta_obs.append(t[n_dim_lat:, n_dim_lat:] - L)
        ells.append(L)
    return thetas, theta_obs, ells


def make_ticc(rand_seed, num_blocks=5, n_dim=10, sparsity_inv_matrix=0.5):
    import networkx as nx
    np.random.seed(rand_seed)
    size_blocks = n_dim
    block_matrices = {}

    def genInvCov(size, low=0.3, upper=0.6, portion=0.2, symmetric=True):
        portion = portion / 2
        S = np.zeros((size, size))
        n_edges = int((size * (size - 1)) * portion)
        G = nx.gnm_random_graph(size, n_edges)
        for src, dest in G.edges:
            S[src, dest] = (np.random.randint(2) - 0.5) * 2 * (
                low + (upper - low) * np.random.rand(1)[0])
        if symmetric:
            S += S.T
        # vals = alg.eigvalsh(S)
        # S = S + (0.1 - vals[0])*np.identity(size)
        return S

    def genRandInv(size, low=0.3, upper=0.6, portion=0.2):
        S = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if np.random.rand() < portion:
                    value = (np.random.randint(2) - 0.5) * 2 * (
                        low + (upper - low) * np.random.rand(1)[0])
                    S[i, j] = value
        return S

    ##Generate all the blocks
    for block in range(num_blocks):
        if block == 0:
            block_matrices[block] = genInvCov(
                size=size_blocks, portion=sparsity_inv_matrix,
                symmetric=(block == 0))
        else:
            block_matrices[block] = genRandInv(
                size=size_blocks, portion=sparsity_inv_matrix)

    ##Initialize the inverse matrix
    inv_matrix = np.zeros([num_blocks * size_blocks, num_blocks * size_blocks])

    ##go through all the blocks
    for block_i in range(num_blocks):
        for block_j in range(num_blocks):
            block_num = np.abs(block_i - block_j)
            inv_matrix[
                block_i * size_blocks:(block_i + 1) * size_blocks,
                block_j * size_blocks:(block_j + 1) * size_blocks
            ] = block_matrices[block_num] if block_i > block_j \
                else np.transpose(block_matrices[block_num])

    ##print out all the eigenvalues
    eigs, _ = np.linalg.eig(inv_matrix)
    lambda_min = min(eigs)

    ##Make the matrix positive definite
    inv_matrix += (0.1 + abs(lambda_min)) * np.eye(size_blocks * num_blocks)

    eigs, _ = np.linalg.eig(inv_matrix)
    lambda_min = min(eigs)
    print("Modified Eigenvalues are:", np.sort(eigs))

    return inv_matrix
