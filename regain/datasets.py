"""Dataset generation module."""
import numpy as np

from sklearn.datasets.base import Bunch


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


def generate_dataset(n_samples=100, n_dim_obs=100, n_dim_lat=10, T=10,
                     mode="evolving", **kwargs):
    """Generate a synthetic dataset.

    Parameters
    ----------
    n_samples: int,
        number of samples to generate
    n_dim_obs: int,
        number of observed variables of the graph
    n_dim_lat: int,
        number of latent variables of the graph
    T: int,
        number of times
    mode: string,
        "evolving": generate a dataset with evolving observed and latent
                    variables that have a small Frobenious norm between two
                    close time points
        "fixed": generate a dataset with evolving observed and fixed latent
                    variables that have a small Frobenious norm between two
                    close time points
        "l1": generate a dataset with evolving observed and latent variables
              that differs for a small l1 norm
        "l1l2": generate a dataset with evolving observed variables that
                differs for a small l1 norm and evolving latent variables
                that differs for a small l2 norm
        "sin": generate a dataset with fixed latent variables and evolving
                observed variables that are generated from sin functions.
    *kwargs: other arguments related to each specific data generation mode

    """
    if mode == "evolving":
        func = generate_dataset_with_evolving_L
    elif mode == "fixed":
        func = generate_dataset_with_fixed_L
    elif mode == "l1":
        func = generate_dataset_L1
    elif mode == "l1l2":
        func = generate_dataset_L1L2
    elif mode == "sin":
        func = generate_dataset_sin_cos
    else:
        ValueError("Unknown mode %s. Choices are: `evolving`, `fixed`, `l1`, "
                   "`l1l2`, `sin`." % mode)
    n_dim_obs = int(n_dim_obs)
    n_dim_lat = int(n_dim_lat)
    n_samples = int(n_samples)

    thetas, thetas_obs, ells = func(n_dim_obs, n_dim_lat, T, **kwargs)
    sigmas = np.array(map(np.linalg.inv, thetas_obs))
    map(normalize_matrix, sigmas)  # in place

    data_list = np.array([np.random.multivariate_normal(
        np.zeros(n_dim_obs), sigma, size=n_samples) for sigma in sigmas])
    return Bunch(data=data_list,
            thetas=np.array(thetas),
            thetas_observed=np.array(thetas_obs),
            ells=np.array(ells))


def generate_dataset_L1L2(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """DESCRIZIONE, PRIMA O POI."""
    degree = kwargs.get('degree', 2)
    proportional = kwargs.get('proportional', False)
    epsilon = kwargs.get('epsilon', 1e-2)

    K_HO = np.zeros((n_dim_lat, n_dim_obs))
    for i in range(n_dim_lat):
        percentage = int(n_dim_obs * 0.8)
        indices = np.random.randint(0, high=n_dim_obs, size=percentage)
        K_HO[i, indices] = np.random.rand(percentage) * (0.12 / (n_dim_obs/100))
    L = K_HO.T.dot(K_HO)
    assert(is_pos_semidef(L))
    assert np.linalg.matrix_rank(L) == n_dim_lat

    theta = np.eye(n_dim_obs)
    for i in range(n_dim_obs):
        l = list(set(np.arange(0, n_dim_obs)) - set().union(
            list(np.nonzero(theta[i, :])[0]),
            list(np.where(np.count_nonzero(theta, axis=1) >= 3)[0])))
        if len(l) == 0:
            continue
        indices = np.random.choice(l, degree - np.count_nonzero(theta[i, :]) + 1)
        theta[i, indices] = theta[indices, i] = .5 / degree
    assert(is_pos_def(theta))
    theta_observed = theta - L
    assert(is_pos_def(theta_observed))

    thetas = [theta]
    thetas_obs = [theta_observed]
    ells = [L]
    K_HOs = [K_HO]

    for i in range(1, T):
        if proportional:
            no = int(np.ceil(n_dim_obs / 20))  # TODO parametrise the 20
        else:
            no = 1

        rows = np.zeros(no)
        cols = np.zeros(no)
        while (np.any(rows == cols)):
            rows = np.random.randint(0, n_dim_obs, no)
            cols = np.random.randint(0, n_dim_obs, no)
        theta = thetas[-1].copy()
        for r, c in zip(rows, cols):
            theta[r, c] = 0.12 if theta[r, c] == 0 else 0
            theta[c, r] = theta[r, c]
        assert(is_pos_def(theta))

        K_HO = K_HOs[-1].copy()
        addition = np.random.rand(*K_HO.shape)
        addition *= (epsilon / np.linalg.norm(addition))
        K_HO += addition
        K_HO = K_HO / np.sum(K_HO, axis=1)[:, None]
        K_HO *= 0.12 / (n_dim_obs/100)
        K_HO[np.abs(K_HO) < epsilon / theta.shape[0]] = 0
        K_HOs.append(K_HO)
        L = K_HO.T.dot(K_HO)
        assert np.linalg.matrix_rank(L) == n_dim_lat
        assert(is_pos_semidef(L))
        assert(is_pos_def(theta - L))
        L = K_HO.T.dot(K_HO)
        assert np.linalg.matrix_rank(L) == n_dim_lat
        assert(is_pos_semidef(L))
        assert(is_pos_def(theta - L))

        thetas.append(theta)
        thetas_obs.append(theta - L)
        ells.append(L)
        K_HOs.append(K_HO)

    return thetas, thetas_obs, ells


def generate_dataset_L1(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """DESCRIZIONE, PRIMA O POI."""
    degree = kwargs.get('degree', 2)
    proportional = kwargs.get('proportional', False)

    K_HO = np.zeros((n_dim_lat, n_dim_obs))
    for i in range(n_dim_lat):
        percentage = int(n_dim_obs * 0.8)
        indices = np.random.randint(0, high=n_dim_obs, size=percentage)
        K_HO[i, indices] = np.random.rand(percentage) * 0.12
    L = K_HO.T.dot(K_HO)
    assert(is_pos_semidef(L))
    assert np.linalg.matrix_rank(L) == n_dim_lat

    theta = np.eye(n_dim_obs)
    for i in range(n_dim_obs):
        l = list(set(np.arange(0, n_dim_obs)) -
                set().union(list(np.nonzero(theta[i, :])[0]),
                            list(np.where(np.count_nonzero(theta, axis=1) >= 3)[0])))
        if len(l) == 0:
            continue
        indices = np.random.choice(l, degree-(np.count_nonzero(theta[i,:]) - 1))
        theta[i, indices] = theta[indices, i] = .5 / degree
    assert(is_pos_def(theta))
    theta_observed = theta - L
    assert(is_pos_def(theta_observed))

    thetas = [theta]
    thetas_obs = [theta_observed]
    ells = [L]
    K_HOs = [K_HO]

    for i in range(1, T):
        if proportional:
            no = int(np.ceil(n_dim_obs/20))
        else:
            no = 1

        rows = np.zeros(no)
        cols = np.zeros(no)
        while (np.any(rows==cols)):
            rows = np.random.randint(0, n_dim_obs, no)
            cols = np.random.randint(0, n_dim_obs, no)
        theta = thetas[-1].copy()
        for r, c in zip(rows, cols):
            theta[r,c] = 0.12 if theta[r,c] == 0 else 0;
            theta[c,r] = theta[r,c]
       # print(theta)
        assert(is_pos_def(theta))

        K_HO = K_HOs[-1].copy()
        c = np.random.randint(0, n_dim_obs, 1)
        r = np.random.randint(0, n_dim_lat, 1)
        K_HO[r,c] = 0.12 if K_HO[r,c] == 0 else 0;
        #K_HO[c,r] = K_HO[r,c]

        L = K_HO.T.dot(K_HO)
        assert np.linalg.matrix_rank(L) == n_dim_lat
        assert(is_pos_semidef(L))
        assert(is_pos_def(theta - L))

        thetas.append(theta)
        thetas_obs.append(theta - L)
        ells.append(L)
        K_HOs.append(K_HO)

    return thetas, thetas_obs, ells


def generate_dataset_with_evolving_L(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """
    descrizione prima o poi"""

    degree= kwargs.get('degree',2)
    epsilon=kwargs.get('epsilon',1e-2)
    K_HO = np.zeros((n_dim_lat, n_dim_obs))
    for i in range(n_dim_lat):
        percentage = int(n_dim_obs * 0.8)
        indices = np.random.randint(0, high=n_dim_obs, size=percentage)
        K_HO[i, indices] = np.random.rand(percentage) * 0.12
    L = K_HO.T.dot(K_HO)
    assert(is_pos_semidef(L))
    assert np.linalg.matrix_rank(L) == n_dim_lat

    theta = np.eye(n_dim_obs)
    for i in range(n_dim_obs):
        l = list(set(np.arange(0, n_dim_obs)) -
                set().union(
                list(np.nonzero(theta[i,:])[0]),
                list(np.where(np.count_nonzero(theta, axis=1)>=3)[0])))
        if len(l)==0: continue
        indices = np.random.choice(l, degree-(np.count_nonzero(theta[i,:])-1))
        theta[i, indices] = theta[indices, i] = .5 / degree
    assert(is_pos_def(theta))
    theta_observed = theta - L
    assert(is_pos_def(theta_observed))

    thetas = [theta]
    thetas_obs = [theta_observed]
    ells = [L]
    K_HOs = [K_HO]

    for i in range(1,T):
        addition = np.zeros(thetas[-1].shape)

        for i in range(theta.shape[0]):
            addition[i, np.random.randint(0, theta.shape[0], size=degree)] = np.random.randn(degree)
        addition[np.triu_indices(theta.shape[0])[::-1]] = addition[np.triu_indices(theta.shape[0])]
        addition *= (epsilon/np.linalg.norm(addition))
        np.fill_diagonal(addition, 0)
        addition *= epsilon / np.linalg.norm(addition)
        theta = thetas[-1] + addition
        theta[np.abs(theta)<2*epsilon/(theta.shape[0])] = 0
        for j in range(n_dim_obs):
            indices = list(np.where(theta[j,:]!=0)[0])
            indices.remove(j)
            if(len(indices)>degree):
                choice = np.random.choice(indices, len(indices)-degree)
                theta[j,choice] = 0
                theta[choice,j] = 0
        # plot_graph_with_latent_variables(theta, 0, theta.shape[0], "Theta" + str(i))
        assert(is_pos_def(theta))

        K_HO = K_HOs[-1].copy()
        addition = np.random.rand(*K_HO.shape)
        addition *= (epsilon / np.linalg.norm(addition))
        K_HO += addition
        K_HO = K_HO / np.sum(K_HO, axis=1)[:, None]
        K_HO *=0.12
        K_HO[np.abs(K_HO)<epsilon/(theta.shape[0])] = 0
        K_HOs.append(K_HO)
        L = K_HO.T.dot(K_HO)
        assert np.linalg.matrix_rank(L) == n_dim_lat
        assert(is_pos_semidef(L))
        assert(is_pos_def(theta - L))
        thetas.append(theta)
        thetas_obs.append(theta - L)
        ells.append(L)
        K_HOs.append(K_HO)

    return thetas, thetas_obs, ells


def generate_dataset_with_fixed_L(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Generate precisions with a fixed L matrix."""

    degree= kwargs.get('degree',2)
    epsilon=kwargs.get('epsilon',1e-2)
    K_HO = np.zeros((n_dim_lat, n_dim_obs))
    for i in range(n_dim_lat):
        percentage = int(n_dim_obs * 0.8)
        indices = np.random.randint(0, high=n_dim_obs, size=percentage)
        K_HO[i, indices] = np.random.rand(percentage)
    L = K_HO.T.dot(K_HO)
    L *= (0.12/np.sqrt(n_dim_obs))/np.max(L)
    assert(is_pos_semidef(L))
    assert np.linalg.matrix_rank(L) == n_dim_lat

    theta = np.eye(n_dim_obs)
    for i in range(n_dim_obs):
        l = list(set(np.arange(0, n_dim_obs)) -
                set().union(list(np.nonzero(theta[i,:])[0]),
                            list(np.where(np.count_nonzero(theta, axis=1)>=3)[0])))
        if len(l) == 0:
            continue
        indices = np.random.choice(l, degree-(np.count_nonzero(theta[i,:])-1))
        theta[i, indices] = theta[indices, i] = .5 / degree
    assert(is_pos_def(theta))
    theta_observed = theta - L
    assert(is_pos_def(theta_observed))

    thetas = [theta]
    thetas_obs = [theta_observed]

    for i in range(1, T):
        addition = np.zeros(thetas[-1].shape)
        for i in range(n_dim_obs):
            addition[i, np.random.randint(0, n_dim_obs, size=degree)] = np.random.randn(degree)
        addition[np.triu_indices(n_dim_obs)[::-1]] = addition[np.triu_indices(n_dim_obs)]
        addition *= (epsilon/np.linalg.norm(addition))
        np.fill_diagonal(addition, 0)
        theta = thetas[-1] + addition
        theta[np.abs(theta)<2*epsilon/(theta.shape[0])] = 0
        #theta[np.abs(theta)<1e-2] = 0
        for j in range(n_dim_obs):
            indices = list(np.where(theta[j,:]!=0)[0])
            indices.remove(j)
            if(len(indices)>degree):
                choice = np.random.choice(indices, len(indices)-degree)
                theta[j,choice] = 0
                theta[choice,j] = 0

        #plot_graph_with_latent_variables(theta, 0, theta.shape[0], "Theta"+str(i))
        assert(is_pos_def(theta))
        assert(is_pos_def(theta - L))
        thetas.append(theta)
        thetas_obs.append(theta - L)

    return thetas, thetas_obs, np.array([L] * T)


def generate_dataset_sin_cos(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Aggiungi descrizione."""
    degree = kwargs.get('sparsity', 2)
    eps = kwargs.get('eps', 1e-2)
    K_HO = np.zeros((n_dim_lat, n_dim_obs))
    for i in range(n_dim_lat):
        percentage = int(n_dim_obs * 0.8)
        indices = np.random.randint(0, high=n_dim_obs, size=percentage)
        K_HO[i, indices] = np.random.rand(percentage)
    L = K_HO.T.dot(K_HO)
    L *= (0.12/np.sqrt(n_dim_obs))/np.max(L)
    assert(is_pos_semidef(L))
    assert np.linalg.matrix_rank(L) == n_dim_lat

    phase = np.random.randn(n_dim_obs, n_dim_obs)*np.pi
    phase[np.triu_indices(n_dim_obs)[::-1]] = phase[np.triu_indices(n_dim_obs)]

    clip = np.zeros((n_dim_obs, n_dim_obs))
    picks = np.random.permutation(len(np.triu_indices(n_dim_obs, 1)[0]))
    dim = int(len(np.triu_indices(n_dim_obs, 1)[0]) * degree)
    picks = picks[:dim]
    clip1 = clip[np.triu_indices(n_dim_obs, 1)].ravel()
    clip1[picks] = 1
    clip[np.triu_indices(n_dim_obs, 1)[::-1]] = clip1
    clip[np.triu_indices(n_dim_obs, 1)] = clip1

    thetas = np.array([np.eye(n_dim_obs) for i in range(T)])

    x = np.linspace(-np.pi, np.pi, T)
    for i in range(T):
        for r in range(thetas[i].shape[0]):
            for c in range(thetas[i].shape[1]):
                if r == c:
                    continue
                if clip[r, c]:
                    thetas[i, r, c] = np.sin((x[i]+phase[r, c])/T**2)*(0.5/T)
                else:
                    thetas[i, r, c] = np.sin((x[i]+phase[r, c]))*(0.5/T)
        thetas[i][clip == 1] = np.clip(thetas[i][clip == 1], 0,1)
        thetas[i][np.abs(thetas[i]) < eps] = 0

        assert(is_pos_def(thetas[i]))
        theta_observed = thetas[i] - L
        assert(is_pos_def(theta_observed))
        thetas_obs = [theta_observed]

    return thetas, thetas_obs, np.array([L]*T)


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


def generate_ma_xue_zou(n_dim_obs=12, n_latent=3, epsilon=1e-3, sparsity=0.1):
    """Generate the dataset as in Ma, Xue, Zou (2012)."""
    # p = n_dim_obs + n_latent  # int(n_dim_obs * 0.05)
    p = n_dim_obs + int(n_dim_obs * 0.05)
    po = n_dim_obs
    ph = p - n_dim_obs
    W = np.zeros((p, p))
    non_zeros = int(round(p*p*sparsity))
    picks = np.random.permutation(p*p)[:non_zeros]
    W = W.ravel(order='F')
    W[picks] = np.random.randn(non_zeros)
    W = np.reshape(W, (p, p), order="F")

    C = W.T.dot(W)
    C[:po, po:] += 0.5 * np.random.randn(po, ph)
    C = (C + C.T) / 2.

    C = np.clip(C - np.diag(np.diag(C)), -1, 1)
    eig, Q = np.linalg.eigh(C)
    K = C + max(-1.2 * np.min(eig), 0.001) * np.eye(p)
    KO = K[:po, :po]
    KOH = K[:po, po:]
    KHO = K[po:, :po]
    KH = K[po:, po:]

    # L = np.divide(KOH, KH.dot(KHO))
    assert np.allclose(KOH, KHO.T)
    L = np.linalg.multi_dot((KOH, np.linalg.inv(KH), KHO))
    KOtilde = KO - L
    assert is_pos_def(KOtilde)
    assert is_pos_semidef(KH)
    assert np.linalg.matrix_rank(L) == ph
    print(ph)

    N = 5 * po * 2
    cov = np.linalg.inv(KOtilde)
    cov = (cov + cov.T) / 2.
    data = np.random.multivariate_normal(np.zeros(po), cov, size=N)
    # emp_cov = 1. / N * data.T.dot(data)
    return data, KOtilde, KO, L
