import numpy as np

from sklearn.utils import check_random_state
from scipy.stats import norm

from regain.utils import is_pos_def


def compute_probabilities(locations, random_state, mask=None):
    p = len(locations)
    if mask is None:
        mask = np.zeros((p, p))
    probabilities = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            d = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
            d = d/10 if mask[i, j] else d
            probabilities[i, j] = 2*norm.pdf(d*np.sqrt(p), scale=1)
    thresholds = random_state.uniform(low=0, high=1, size=(p, p))
    probabilities[np.where(thresholds > probabilities)] = 0
    return probabilities


def data_Meinshausen_Yuan(p=198, h=2, n=200, random_state=None):
    random_state = check_random_state(random_state)
    nodes_locations = []
    for i in range(p):
        nodes_locations.append(list(random_state.uniform(size=2)))
    probs = compute_probabilities(nodes_locations, random_state)
    theta = np.zeros((p, p))
    for i in range(p-1):
        nz = list(np.nonzero(probs[i, :])[0])
        nz = np.array(list(set(nz) - set(np.arange(0, i))))
        np.random.shuffle(nz)
        no_nonzero = np.sum((theta != 0).astype(int), axis=1)[i]
        if no_nonzero >= 4 or len(nz) == 0:
            continue
        theta[i, nz[:max(0, 4 - no_nonzero)]] = 0.245
        theta[nz[:max(0, 4 - no_nonzero)], i] = 0.245
    np.fill_diagonal(theta, 1)
    assert is_pos_def(theta)

    latent = np.zeros((h, p))
    for i in range(h):
        for j in range(p):
            latent[i, j] = random_state.uniform(low=0, high=0.1, size=1)

    L = latent.T.dot(latent)
    theta_observed = theta - L
    sigma = np.linalg.inv(theta_observed)
    assert is_pos_def(sigma)

    samples = random_state.multivariate_normal(np.zeros(p), sigma, size=n)
    return nodes_locations, theta, theta_observed, sigma, latent, samples
