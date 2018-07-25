import numpy as np


def lognpdf(x, mu, sigma):
    """This is like Matlab and numpy. stats.lognorm.pdf is different for the mean."""
    return np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2) \
        / (x * sigma * np.sqrt(2 * np.pi))


def lognstat(m, v):
    m2 = m * m
    mu = np.log(m2 / np.sqrt(v + m2))
    sigma = np.sqrt(np.log(v / m2 + 1))
    return mu, sigma
