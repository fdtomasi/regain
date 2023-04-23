import numpy as np


def batch_logdet(A) -> np.ndarray:
    """Equivalent to sklearn.extmath.fast_logdet
    but supports multiple matrices natively.
    """
    sign, ld = np.linalg.slogdet(A)
    return np.where(sign > 0, ld, np.full_like(ld, -np.inf))


def fill_diagonal(a, diag):
    np.einsum("...jj->...j", a)[...] = diag


def get_diagonal(a):
    """Return diagonal(s) from possibly multi dimensional tensor."""
    return np.diagonal(a, axis1=-2, axis2=-1)


def create_from_diagonal(x):
    """Diagonal matrix with support for ndim > 2 tensors."""
    n_dim = x.shape[-1]
    batch_dims = x.shape[:-1]
    a = np.zeros((*batch_dims, n_dim, n_dim))
    fill_diagonal(a, x)
    return a
