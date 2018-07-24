import numpy as np
from scipy import stats
from sklearn.utils.extmath import squared_norm


def GWP_construct(umat, L):
    v, p, n = umat.shape
    M = np.zeros((p, p, n))

    for i in range(n):
        for j in range(v):
            Lu = L.dot(umat[j, :, i])
            LuuL = Lu[:, None].dot(Lu[None, :])
            M[..., i] += LuuL

    # Lu =
    # N = np.array([np.linalg.multi_dot((L, umat[..., i].T, umat[..., i], L.T)) for i in range(n)]).transpose()
    return M


def log_lik_frob(S, D, sigma2):
    """Frobenius norm log likelihood."""
    logl = -0.5 * (S.size * np.log(2. * np.pi * sigma2)
                   + squared_norm(S - D) / sigma2)
    return logl


def log_likelihood_normal(x, mean, var):
    """Normal log likelihood."""
    logl = -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)
    # logl2 = stats.norm.logpdf(x, loc=mean, scale=np.sqrt(var))
    # assert logl == logl2, (logl, logl2)
    return logl
