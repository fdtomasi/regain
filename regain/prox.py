"""Useful proximal functions."""
import numpy as np
import warnings


def soft_thresholding(a, lamda):
    """Soft-thresholding for vectors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.maximum(0, 1 - lamda / np.linalg.norm(a)) * a


def soft_thresholding_sign(a, lamda):
    """Soft-thresholding."""
    return np.sign(a) * np.maximum(np.abs(a) - lamda, 0)


def soft_thresholding_od(a, lamda):
    """Off-diagonal soft-thresholding."""
    soft = np.sign(a) * np.maximum(np.abs(a) - lamda, 0)
    soft.flat[::a.shape[1] + 1] = np.diag(a)
    return soft


def prox_logdet(A, lamda):
    """Time-varying latent variable graphical lasso prox."""
    es, Q = np.linalg.eigh(A)
    xi = (- es + np.sqrt(np.square(es) + 4. / lamda)) * lamda / 2.
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))


def prox_trace_indicator(A, lamda):
    """Time-varying latent variable graphical lasso prox."""
    es, Q = np.linalg.eigh(A)
    xi = np.maximum(es - lamda, 0)
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))


def prox_laplacian(A, beta):
    """Prox for l_2 square norm, Laplacian regularisation."""
    return A / (1 + 2. * beta)
