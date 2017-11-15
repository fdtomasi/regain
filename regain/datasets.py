"""Dataset generation module."""
import numpy as np
import sys
import scipy.sparse as ss

from sklearn.datasets import make_sparse_spd_matrix


def normalize_matrix(x):
    """Normalize a matrix so to have 1 on the diagonal, in-place."""
    d = np.diag(x).reshape(1, x.shape[0])
    d = 1. / np.sqrt(d)
    x *= d
    x *= d.T


def is_pos_def(x, tol=1e-15):
    """Check if x is positive definite."""
    eigs = np.linalg.eigvalsh(x)
    eigs[np.abs(eigs) < tol] = 0
    return np.all(eigs > 0)


def is_pos_semidef(x, tol=1e-15):
    """Check if x is positive semi-definite."""
    eigs = np.linalg.eigvalsh(x)
    eigs[np.abs(eigs) < tol] = 0
    return np.all(eigs >= 0)


def generate(n_dim_obs=3, n_dim_lat=2, eps=1e-3, T=10, tol=1e-3):
    theta = make_sparse_spd_matrix(n_dim_lat + n_dim_obs, alpha=.3, norm_diag=1)

    theta[n_dim_lat:, n_dim_lat:] = make_sparse_spd_matrix(
        n_dim_obs, alpha=.5, norm_diag=1)

#     theta_tot = theta_tot.dot(theta_tot.T)
#     assert is_pos_def(theta_tot)
    assert np.linalg.matrix_rank(theta[:n_dim_lat, :n_dim_lat]) == n_dim_lat

    thetas = [theta]
    es, Q = np.linalg.eigh(theta[n_dim_lat:, n_dim_lat:])

    for i in range(T):
        # theta_tot_i = make_sparse_spd_matrix(
        #     n_dim_lat + n_dim_obs, alpha=.5, norm_diag=1)
        # theta_tot_i.flat[::n_dim_lat + n_dim_obs + 1] = 1e-16
        # theta_tot_i /= np.linalg.norm(theta_tot_i, 'fro')
        # theta_tot_i *= eps
        es += (np.random.randn(es.shape[0]) + 1) / 2.
        theta_i = np.linalg.multi_dot((Q, np.diag(es), Q.T))

        theta_c = theta.copy()
        theta_c[n_dim_lat:, n_dim_lat:] = theta_i
        # theta += theta_tot_i
        # theta.flat[::n_dim_lat + n_dim_obs + 1] = 0
        # theta /= np.amax(np.abs(theta))
        # # min((np.linalg.norm(theta2, 'fro')/((n_dim_lat + n_dim_obs)**2)), 0.99)
        # threshold = eps / np.power(n_dim_lat + n_dim_obs, 2)
        # theta[np.abs(theta) < threshold] = 0
        # theta[np.abs(theta) < tol] = 0
        # theta.flat[::n_dim_lat + n_dim_obs + 1] = 1
        # assert is_pos_def(theta)
        thetas.append(theta_c)
    return np.array(thetas)


def generate2(n_dim_obs=3, n_dim_lat=2, epsilon=1e-3, T=10, degree=2):
    K_HO = np.zeros((n_dim_lat, n_dim_obs))
    for i in range(n_dim_lat):
        percentage = int(n_dim_obs * 0.8)
        indices = np.random.randint(0, high=n_dim_obs, size=percentage)
        K_HO[i, indices] = np.random.rand(percentage)  # *0.12

    eigs, U = np.linalg.eigh(K_HO.T.dot(K_HO))
    eigs = np.zeros(n_dim_obs)
    eigs[np.random.randint(0, n_dim_obs, size=n_dim_lat)] = np.random.rand(n_dim_lat)
    L = np.linalg.multi_dot((U, np.diag(eigs), U.T))

    theta = np.eye(n_dim_obs)
    for i in range(n_dim_obs - 1):
        l = list(np.arange(i + 1, n_dim_obs))
        indices = np.random.choice(l, degree)
        theta[i, indices] = theta[indices, i] = .5 / degree

    #theta += np.diag(np.sum(L, axis=1))
    theta_observed = theta - L
    #print(theta_ob)
    thetas = [theta]
    thetas_obs = [theta_observed]
    ells = [L]

    for i in range(T):
        eigs, Q = np.linalg.eigh(thetas[-1])
        addition = ss.rand(eigs.shape[0], 1, density=0.5).A
        addition /= np.linalg.norm(addition) * np.sqrt(epsilon)
        s_new = (eigs + addition.T)[0]
        s_new = np.maximum(s_new, 0)
        #

        theta = np.linalg.multi_dot((Q, np.diag(s_new), Q.T))
        normalize_matrix(theta)
        theta[np.where(np.abs(theta) <1e-2)] = 0
        print(theta)
        #

        eigs, Q = np.linalg.eigh(ells[-1])
        addition = ss.rand(eigs.shape[0], 1, density=0.5).A
        addition /= np.linalg.norm(addition) * np.sqrt(epsilon)
        s_new = (eigs + addition.T)[0]
        s_new = np.maximum(s_new, 0)

        L = np.linalg.multi_dot((Q, np.diag(s_new), Q.T))
        normalize_matrix(L)

        theta += np.diag(np.sum(np.abs(L), axis=1))
        theta_observed = theta - L
        #theta_observed[np.where(np.abs(theta_observed) <1e-2)] = 0
        assert is_pos_semidef(L)
        assert is_pos_def(theta_observed)
        thetas.append(theta)
        thetas_obs.append(theta_observed)
        ells.append(L)

    return thetas, thetas_obs, ells

def generate_dataset_with_fixed_L(n_dim_obs=10, n_dim_lat=2, epsilon=1e-3,
                                  T=10, degree=2):

    K_HO = np.zeros((n_dim_lat, n_dim_obs))
    for i in range(n_dim_lat):
        percentage = int(n_dim_obs * 0.8)
        indices = np.random.randint(0, high=n_dim_obs, size=percentage)
        K_HO[i, indices] = np.random.rand(percentage)*0.12
    L = K_HO.T.dot(K_HO)
    assert(is_pos_semidef(L))

    theta = np.eye(n_dim_obs)
    for i in range(n_dim_obs - 1):
        l = list(np.arange(i + 1, n_dim_obs))
        indices = np.random.choice(l, degree)
        theta[i, indices] = theta[indices, i] = .5 / degree

    assert(is_pos_def(theta))
    theta_observed = theta - L
    assert(is_pos_def(theta_observed))

    thetas = [theta]
    thetas_obs = [theta_observed]

    for i in range(1,T):
        addition = np.zeros(thetas[-1].shape)
        for i in range(theta.shape[0]):
            addition[i, np.random.randint(0, theta.shape[0], size=degree)] = np.random.randn(degree)
        addition[np.triu_indices(theta.shape[0])[::-1]] = addition[np.triu_indices(theta.shape[0])]
        addition *= (epsilon/np.linalg.norm(addition))
        np.fill_diagonal(addition, 0)
        theta = thetas[-1] + addition
        theta[np.abs(theta)<2*epsilon/(theta.shape[0])] = 0
        #plot_graph_with_latent_variables(theta, 0, theta.shape[0],
        #                                 "Theta" + str(i))
        assert(is_pos_def(theta))
        assert(is_pos_def(theta-L))
        thetas.append(theta)
        thetas_obs.append(theta-L)

    return thetas, thetas_obs, L

def generate_dataset(n_dim_obs=3, n_dim_lat=2, eps=1e-3, T=10,
                     n_samples=50):
    thetas = generate(n_dim_obs, n_dim_lat, eps, T)
    sigmas = np.array(map(np.linalg.inv, thetas))
    data_list = []
    for sigma in sigmas:
        data_list.append(np.random.multivariate_normal(
            np.zeros(n_dim_obs), sigma[n_dim_lat:, n_dim_lat:], n_samples))
    return data_list, thetas, np.array([t[n_dim_lat:, n_dim_lat:] for t in thetas])


def generate_dataset_fede(
        n_dim_obs=3, n_dim_lat=2, eps=1e-3, T=10, n_samples=50):
    """Generate dataset (new version)."""
    b = np.random.rand(1, n_dim_obs)
    es, Q = np.linalg.eigh(b.T.dot(b))  # Q random

    b = np.random.rand(1, n_dim_obs)
    es, R = np.linalg.eigh(b.T.dot(b))  # R random

    start_sigma = np.random.rand(n_dim_obs) + 1
    start_lamda = np.zeros(n_dim_obs)
    idx = np.random.randint(n_dim_obs, size=n_dim_lat)
    start_lamda[idx] = np.random.rand(2)

    Ks = []
    Ls = []
    Kobs = []

    for i in range(T):
        K = np.linalg.multi_dot((Q, np.diag(start_sigma), Q.T))
        L = np.linalg.multi_dot((R, np.diag(start_lamda), R.T))

        K[np.abs(K) < eps] = 0  # enforce sparsity on K

        assert is_pos_def(K - L)
        assert is_pos_semidef(L)

        start_sigma += 1 + np.random.rand(n_dim_obs)
        start_lamda[idx] += np.random.rand(n_dim_lat) * 2 - 1
        start_lamda = np.maximum(start_lamda, 0)

        Ks.append(K)
        Ls.append(L)
        Kobs.append(K - L)

    ll = map(np.linalg.inv, Kobs)
    map(normalize_matrix, ll)  # in place

    data_list = [np.random.multivariate_normal(
        np.zeros(n_dim_obs), l, size=n_samples) for l in ll]
    return data_list, Kobs, Ks, Ls


def generate_ma_xue_zou(n_dim_obs=12, n_dim_lat=2, epsilon=1e-3):
    """Generate the dataset as in Ma, Xue, Zou (2012)."""
    theta = make_sparse_spd_matrix(n_dim_lat + n_dim_obs, alpha=.9, norm_diag=1)
    theta_flat = theta.flatten()
    idx = (theta_flat != 0)

    proba = np.random.randn(*theta_flat.shape)
    proba = np.where(proba > 0, 1, -1)

    theta_flat[idx] = proba[idx]

    U = theta_flat.reshape(theta.shape)
    UTU = U.dot(U.T)
#     UTU.flat[::theta.shape[0]+1] = 1
    try:
        K = np.linalg.inv(UTU)
        L = K[n_dim_lat:, 0:n_dim_lat].dot(K[0:n_dim_lat, 0:n_dim_lat]).dot(
            K[0:n_dim_lat, n_dim_lat:])
        assert (is_pos_semidef(L))
        assert (is_pos_def(K[n_dim_lat:, n_dim_lat:] - L))

    except:
        sys.stdout.write("-")
