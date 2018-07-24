import numpy as np


def lognpdf(x, mu, sigma):
    """This is like Matlab and numpy. stats.lognorm.pdf is different for the mean."""
    return np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2) \
        / (x * sigma * np.sqrt(2 * np.pi))


def lognstat(m, v):
    mu = np.log(m**2 / np.sqrt(v + m**2))
    sigma = np.sqrt(np.log(v / m**2 + 1))
    return mu, sigma
