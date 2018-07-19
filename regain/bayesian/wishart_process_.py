import numpy as np
from sklearn.utils.extmath import squared_norm


def GWP_construct(umat, L, nu):
    p = L.shape[1]
    N = umat.shape[2]
    M = np.zeros((p, p, N))

    for i in range(N):
        for j in range(nu):
            Lu = L.dot(umat[j,:,i])
            M[:, :, i] += Lu.dot(Lu.T)
    return M


def log_lik_frob(S, D, sigma2):
    dim = S[:, :, 0].size

    return sum(
        -dim / 2. * np.log(2. * np.pi * sigma2)
        - (.5 / sigma2) * squared_norm(S[:, :, n] - D[:, :, n])
        for n in range(S.shape[-1]))
