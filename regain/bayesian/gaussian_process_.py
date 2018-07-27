import numpy as np


def sample(K, nu=1, p=3):
    """Sample from a GP.

    Parameters
    ----------
    K : ndarray, shape (n, n)
        Temporal kernel between n time points.
    nu : type
        Number of replicates.
    p : type
        Number of time series to generate.

    Returns
    -------
    u : ndarray, shape (nu, p, K.shape[0])
        Samples from a Gaussian process.

    """
    u = np.empty((nu, p, K.shape[0]))
    for j in range(nu):
        u[j] = np.random.multivariate_normal(np.zeros(K.shape[0]), K, size=p)
    return u
