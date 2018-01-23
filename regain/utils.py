"""Utils for REGAIN package."""
import functools
import numpy as np
from collections import namedtuple

convergence = namedtuple('convergence',
                         ('obj', 'rnorm', 'snorm', 'e_pri', 'e_dual'))


def flatten(lst):
    """Flatten a list."""
    return [y for l in lst for y in flatten(l)] \
        if isinstance(lst, (list, np.ndarray)) else [lst]


def upper_to_full(a):
    """Convert the upper part to a full symmetric matrix."""
    n = int((np.sqrt(1 + 8*a.shape[0]) - 1) / 2)
    A = np.zeros((n, n))
    idx = np.triu_indices(n)
    A[idx] = A[idx[::-1]] = a
    return A


def compose(*functions):
    """Compose two or more functions."""
    def compose2(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose2, functions, lambda x: x)


def error_rank(ells_true, ells_pred):
    """Compute the mean absolute error in rank between two matrices.

    Parameters
    ----------
    ells_{true, pred} : array-like, 3 dimensional
        Latent variable matrices for which to compare the rank.

    """
    ranks_true = np.array([np.linalg.matrix_rank(l) for l in ells_true])
    ranks_pred = np.array([np.linalg.matrix_rank(l) for l in ells_pred])
    return np.mean(np.abs(ranks_true - ranks_pred))


def error_norm(cov, comp_cov, norm='frobenius', scaling=True,
               squared=True):
    """Mean Squared Error between two covariance estimators.

    Parameters
    ----------
    cov, comp_cov : array-like, shape = [n_features, n_features]
        The covariance to compare with.

    norm : str
        The type of norm used to compute the error. Available error types:
        - 'frobenius' (default): sqrt(tr(A^t.A))
        - 'spectral': sqrt(max(eigenvalues(A^t.A))
        where A is the error ``(comp_cov - self.covariance_)``.

    scaling : bool
        If True (default), the squared error norm is divided by n_features.
        If False, the squared error norm is not rescaled.

    squared : bool
        Whether to compute the squared error norm or the error norm.
        If True (default), the squared error norm is returned.
        If False, the error norm is returned.

    Returns
    -------
    The Mean Squared Error (in the sense of the Frobenius norm) between
    `self` and `comp_cov` covariance estimators.

    """
    # compute the error
    error = comp_cov - cov
    # compute the error norm
    if norm == "frobenius":
        squared_norm = np.sum(error ** 2)
    elif norm == "spectral":
        squared_norm = np.amax(np.linalg.svdvals(np.dot(error.T, error)))
    else:
        raise NotImplementedError(
            "Only spectral and frobenius norms are implemented")
    # optionally scale the error norm
    if scaling:
        squared_norm = squared_norm / error.shape[0]
    # finally get either the squared norm or the norm
    if squared:
        result = squared_norm
    else:
        result = np.sqrt(squared_norm)
    return result


def error_norm_time(cov, comp_cov, norm='frobenius', scaling=True,
                    squared=True):
    """Mean Squared Error between two covariance estimators.

    Parameters
    ----------
    cov, comp_cov : array-like, shape = [n_time, n_features, n_features]
        The covariance to compare with.

    norm : str
        The type of norm used to compute the error. Available error types:
        - 'frobenius' (default): sqrt(tr(A^t.A))
        - 'spectral': sqrt(max(eigenvalues(A^t.A))
        where A is the error ``(comp_cov - self.covariance_)``.

    scaling : bool
        If True (default), the squared error norm is divided by n_features.
        If False, the squared error norm is not rescaled.

    squared : bool
        Whether to compute the squared error norm or the error norm.
        If True (default), the squared error norm is returned.
        If False, the error norm is returned.

    Returns
    -------
    The Mean Squared Error (in the sense of the Frobenius norm) between
    `self` and `comp_cov` covariance estimators.

    """
    return np.mean([error_norm(
        x, y, norm=norm, scaling=scaling, squared=squared) for x, y in zip(
            cov, comp_cov)])


def structure_error(true, pred, thresholding=False, eps=1e-2):
    """Error in structure between a precision matrix and predicted.

    Parameters
    ----------
    true: array-like
        True matrix. In grpahical inference, if an entry is different from 0
        it is consider as an edge (inverse covariance).

    pred: array-like, shape=(d,d)
        Predicted matrix. In grpahical inference, if an entry is different
        from 0 it is consider as an edge (inverse covariance).

    thresholding: bool, default False,
       Apply a threshold (with eps) to the `pred` matrix.

    eps : float, default = 1e-2
        Apply a threshold (with eps) to the `pred` matrix.

    """
    # avoid inplace modifications
    true = true.copy()
    pred = pred.copy()
    if thresholding:
        pred[np.abs(pred) < eps] = 0
    true[true != 0] = 1
    pred[pred != 0] = 2
    res = true + pred
    FN = np.count_nonzero((res == 1).astype(float))
    FP = np.count_nonzero((res == 2).astype(float))
    TP = np.count_nonzero((res == 3).astype(float))
    TN = np.count_nonzero((res == 0).astype(float))
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall)}
