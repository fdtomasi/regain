"""Utils for REGAIN package."""
from __future__ import division

import functools
import logging
import os
import sys
from collections import namedtuple
from contextlib import contextmanager

import numpy as np
import six
from six.moves import cPickle as pkl

convergence = namedtuple('convergence',
                         ('obj', 'rnorm', 'snorm', 'e_pri', 'e_dual'))


@contextmanager
def suppress_stdout():
    """Suppress function output.

    Usage
    -----
    with suppress_stdout():
        function_with_outputs()
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def _ensure_filename_ending(filename, possible_extensions=['.txt']):
    if isinstance(possible_extensions, six.string_types):
        possible_extensions = [possible_extensions]

    return filename + (
        '' if any(filename.endswith(end) for end in possible_extensions)
        else possible_extensions[0])


def init_logger(filename, verbose=True):
    """Initialise logger."""
    logfile = _ensure_filename_ending(filename, ['.log', '.txt'])
    logging.shutdown()
    root_logger = logging.getLogger()
    for _ in list(root_logger.handlers):
        root_logger.removeHandler(_)
        _.flush()
        _.close()
    for _ in list(root_logger.filters):
        root_logger.removeFilter(_)
        _.flush()
        _.close()

    logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w',
                        format='%(levelname)s (%(asctime)-15s): %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO if verbose else logging.ERROR)
    stream_handler.setFormatter(
        logging.Formatter('%(levelname)s (%(asctime)-15s): %(message)s'))

    root_logger.addHandler(stream_handler)
    return logfile


def save_pickle(obj, filename):
    """Save pickle utility."""
    filename = _ensure_filename_ending(filename, '.pkl')
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)


def load_pickle(filename):
    """Read pickle utility."""
    with open(filename, 'rb') as f:
        res = pkl.load(f)
    return res


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


def structure_error(true, pred, thresholding=False, eps=1e-2,
                    no_diagonal=False):
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
    if no_diagonal:
        if true.ndim > 2:
            true = np.array([t - np.diag(t) for t in true])
            pred = np.array([t - np.diag(t) for t in pred])
        else:
            true -= np.diag(np.diag(true))
            pred -= np.diag(np.diag(pred))
    true[true != 0] = 1
    pred[pred != 0] = 2
    res = true + pred
    TN = np.count_nonzero((res == 0).astype(float))
    FN = np.count_nonzero((res == 1).astype(float))
    FP = np.count_nonzero((res == 2).astype(float))
    TP = np.count_nonzero((res == 3).astype(float))

    precision = TP / float(TP + FP) if TP + FP > 0 else 0
    recall = TP / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    accuracy = (TP + TN) / true.size
    balanced_accuracy = 0.5 * (TP / (TP + FN) + TN / (TN + FP))
    prevalence = (TP + FN) / true.size

    miss_rate = FN / (TP + FN)
    fall_out = FP / (FP + TN)
    specificity = TN / (FP + TN)
    false_discovery_rate = FP / (TP + FP) if TP + FP > 0 else 0
    false_omission_rate = FN / (FN + TN) if FN + TN > 0 else 0
    negative_predicted_value = TN / (FN + TN) if FN + TN > 0 else 0

    positive_likelihood_ratio = recall / fall_out if fall_out > 0 else 0
    negative_likelihood_ratio = miss_rate / specificity if specificity > 0 else 0
    diagnostic_odds_ratio = positive_likelihood_ratio / negative_likelihood_ratio if negative_likelihood_ratio > 0 else 0

    dictionary = dict(
        tp=TP, tn=TN, fp=FP, fn=FN, precision=precision, recall=recall,
        f1=f1, accuracy=accuracy, false_omission_rate=false_omission_rate,
        fdr=false_discovery_rate, npv=negative_predicted_value,
        prevalence=prevalence, miss_rate=miss_rate, fall_out=fall_out,
        specificity=specificity, plr=positive_likelihood_ratio,
        nlr=negative_likelihood_ratio, dor=diagnostic_odds_ratio,
        balanced_accuracy=balanced_accuracy)
    return dictionary
