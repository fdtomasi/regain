import numpy as np

from sklearn.utils import check_random_state

from regain.datasets.single_time import compute_probabilities
from regain.utils import is_pos_def


def permute_locations(locations, random_state):
    new_locations = []
    for l in locations:
        perm = random_state.uniform(low=-0.03, high=0.03, size=(1, 2))
        new_locations.append(np.maximum(
                                        np.array(l) + np.array(perm),
                                        0).flatten().tolist())
    return new_locations


def data_Meinshausen_Yuan(p=198, h=2, n=200, T=10, random_state=None):
    random_state = check_random_state(random_state)
    nodes_locations = []
    for i in range(p):
        nodes_locations.append(list(random_state.uniform(size=2)))
    probs = compute_probabilities(nodes_locations, random_state)
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
        theta[i, ixs] = 0.245
        theta[ixs, i] = 0.245
    to_remove = np.where(np.sum((theta != 0).astype(int), axis=1) > 4)[0]
    for t in to_remove:
        edges = np.nonzero(theta[t, :])[0]
        random_state.shuffle(edges)
        removable = edges[:-4]
        theta[t, removable] = 0
        theta[removable, t] = 0
    np.fill_diagonal(theta, 1)

    assert is_pos_def(theta)
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
    for t in range(T-1):
        theta_prev = thetas[t]
        locs_prev = locations[t]
        locs = permute_locations(locs_prev, random_state)

        locations.append(locs)
        probs = compute_probabilities(locs, random_state, theta_prev != 0)

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
            theta[i, ixs] = 0.245
            theta[ixs, i] = 0.245
        to_remove = np.where(np.sum((theta != 0).astype(int), axis=1) > 4)[0]
        for t in to_remove:
            edges = np.nonzero(theta[t, :])[0]
            random_state.shuffle(edges)
            removable = edges[:-4]
            theta[t, removable] = 0
            theta[removable, t] = 0
        np.fill_diagonal(theta, 1)

        assert is_pos_def(theta)
        thetas.append(theta)
        theta_observed = theta - L
        assert is_pos_def(theta_observed)
        sigma = np.linalg.inv(theta_observed)
        assert is_pos_def(sigma)
        thetas_observed.append(theta_observed)
        covariances.append(sigma)
        sampless.append(random_state.multivariate_normal(np.zeros(p), sigma,
                        size=n))

    return locations, thetas, thetas_observed, covariances, latent, sampless


def data_Meinshausen_Yuan_sparse_latent(p=198, h=2, n=200, T=10,
                                        random_state=None):
    random_state = check_random_state(random_state)
    nodes_locations = []
    for i in range(p+h):
        nodes_locations.append(list(random_state.uniform(size=2)))
    probs = compute_probabilities(nodes_locations, random_state)
    theta = np.zeros((p, p))
    for i in range(p):
        ix = np.argsort(probs[i, :-h])[::-1]
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
        theta[i, ixs] = 0.245
        theta[ixs, i] = 0.245
    to_remove = np.where(np.sum((theta != 0).astype(int), axis=1) > 4)[0]
    for t in to_remove:
        edges = np.nonzero(theta[t, :])[0]
        random_state.shuffle(edges)
        removable = edges[:-4]
        theta[t, removable] = 0
        theta[removable, t] = 0
    np.fill_diagonal(theta, 1)

    assert is_pos_def(theta)

    latent = np.zeros((h, p+h))
    to_put = (p + h)*0.5
    value = 0.98/to_put
    for i in range(h):
        pb = probs[p+i, :]
        ix = np.argsort(pb)[::-1]

        no_nonzero = np.sum((latent != 0).astype(int), axis=1)[i]
        ixs = []
        sel = 1
        j = 0
        while sel <= min(to_put - no_nonzero, np.nonzero(pb)[0].shape[0]) and j < h+h:
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
    theta_obs = theta - connections.T.dot(
                    np.linalg.inv(theta_H)).dot(
                        connections)
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
    for t in range(T-1):
        theta_prev = thetas_O[t]
        lat_prev = thetas_H[t]
        locs_prev = locations[t]
        locs = permute_locations(locs_prev, random_state)

        locations.append(locs)
        probs_theta = compute_probabilities(locs[:p], random_state,
                                            theta_prev != 0)
        probs_lat = compute_probabilities(locs[-h:], random_state,
                                          lat_prev != 0)

        theta = np.zeros((p, p))
        for i in range(p):
            ix = np.argsort(probs_theta[i, :])[::-1]
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
            theta[i, ixs] = 0.245
            theta[ixs, i] = 0.245
        to_remove = np.where(np.sum((theta != 0).astype(int), axis=1) > 4)[0]
        for t in to_remove:
            edges = np.nonzero(theta[t, :])[0]
            random_state.shuffle(edges)
            removable = edges[:-4]
            theta[t, removable] = 0
            theta[removable, t] = 0
        np.fill_diagonal(theta, 1)
        assert is_pos_def(theta)
        thetas_O.append(theta)

        lat = np.zeros((h, h))
        to_put = [np.nonzero(lat_prev[i, :])[0].shape[0] for i in range(h)]

        for i in range(h):
            ix = np.argsort(probs_lat[i, :])[::-1]
            no_nonzero = np.sum((lat != 0).astype(int), axis=1)[i]
            ixs = []
            sel = 1
            j = 0
            while (sel <= to_put[i] - no_nonzero
                    and j < h):
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
        sampless.append(random_state.multivariate_normal(np.zeros(p), sigma,
                        size=n))

    return locations, thetas_O, thetas_H, thetas_obs, connections,sigmas, sampless
