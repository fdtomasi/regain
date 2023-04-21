import numpy as np


def fast_logdet(A):
    """Equivalent to sklearn.extmath.fast_logdet
    but supports multiple matrices natively.
    """
    sign, ld = np.linalg.slogdet(A)
    return np.where(sign > 0, ld, np.full_like(ld, -np.inf))
