# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Utils for REGAIN package."""
from __future__ import division

import collections
import functools
import logging
import os
import sys
import warnings
from contextlib import contextmanager

import numpy as np
import six
from numpy.linalg.linalg import LinAlgError
from scipy import stats
from scipy.spatial.distance import squareform
from six.moves import cPickle as pkl
from sklearn.metrics import average_precision_score, matthews_corrcoef


def display_topics(H, W, feature_names, documents, n_top_words, n_top_documents, print_docs=True):
    """Display topics of LDA."""
    topics = []
    for topic_idx, topic in enumerate(H):
        topics.append(
            " ".join([feature_names[i] + " (%.3f)" % topic[i] for i in topic.argsort()[: -n_top_words - 1 : -1]])
        )

        print("Topic %d: %s" % (topic_idx, topics[-1]))
        top_doc_indices = np.argsort(W[:, topic_idx])[::-1][:n_top_documents]
        if print_docs:
            for i, doc_index in enumerate(top_doc_indices):
                print("doc %d: %s" % (doc_index, documents[doc_index]))
    return topics


def logentropy_normalize(X):
    """Log-entropy normalisation."""
    P = X / X.values.sum(axis=0, keepdims=True)
    E = 1 + (P * np.log(P)).fillna(0).values.sum(axis=0, keepdims=True) / np.log1p(X.shape[0])
    return E * np.log1p(X)


def top_n_indexes(arr, n):
    import bottleneck as bn

    idx = bn.argpartition(arr, arr.size - n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


def retain_top_n(arr, n):
    """Discard low values in a matrix."""
    mask_array = np.zeros_like(arr)
    for idx in top_n_indexes(np.abs(arr), n):
        mask_array[idx] = arr[idx]
    return mask_array


def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


convergence = namedtuple_with_defaults("convergence", "obj rnorm snorm e_pri e_dual precision")


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


def _ensure_filename_ending(filename, possible_extensions=".txt"):
    if isinstance(possible_extensions, six.string_types):
        possible_extensions = [possible_extensions]

    return filename + ("" if any(filename.endswith(end) for end in possible_extensions) else possible_extensions[0])


def init_logger(filename, verbose=True):
    """Initialise logger."""
    logfile = _ensure_filename_ending(filename, [".log", ".txt"])
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

    logging.basicConfig(
        filename=logfile, level=logging.INFO, filemode="w", format="%(levelname)s (%(asctime)-15s): %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO if verbose else logging.ERROR)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s (%(asctime)-15s): %(message)s"))

    root_logger.addHandler(stream_handler)
    return logfile


def save_pickle(obj, filename):
    """Save pickle utility."""
    filename = _ensure_filename_ending(filename, ".pkl")
    with open(filename, "wb") as f:
        pkl.dump(obj, f)


def load_pickle(filename):
    """Read pickle utility."""
    with open(filename, "rb") as f:
        res = pkl.load(f)
    return res


def write_network(dataframe, filename):
    """Write a network as a list of interactions."""
    dataframe.stack().to_csv(filename)


def read_network(filename, threshold=1.0, full_network=True, fill_diagonal=True, delimiter="auto"):
    """Read a network from a list of interactions.

    Parameters
    ----------
    filename : str
        Filename to read from.
    threshold : float, optional
        Only get the top threshold edges.
    full_network : boolean, optional
        Choose if the network is written in full or only the upper triangular.
    fill_diagonal : boolean, optional
        Fill diagonal with the sum of absolute values in the row (for the
        positive definite constraint).
    """
    import pandas as pd

    if filename.endswith(".tsv") or filename.endswith(".tab"):
        if delimiter == "auto":
            delimiter = "\t"
        elif delimiter != "\t":
            warnings.warn(
                "The extension is suggesting the filename is tab-"
                "separated. Please check you are using the correct "
                "separator."
            )
    elif filename.endswith(".csv"):
        if delimiter == "auto":
            delimiter = ","
        elif delimiter != ",":
            warnings.warn(
                "The extension is suggesting the filename is comma-"
                "separated. Please check you are using the correct "
                "separator."
            )
    else:
        if delimiter == "auto":
            raise ValueError("Unrecognized format. Please specify a separator")

    nn = pd.read_csv(filename, delimiter=delimiter, header=None)
    # the following suppose genes are in the form G1, ... G10
    columns = sorted(nn[0].unique(), key=lambda x: int(x[1:]))
    n_top_edges = int(nn.shape[0] * threshold / (2.0 if full_network else 1))

    nn = nn.sort_values(2, ascending=False)[: (2 if full_network else 1) * n_top_edges]

    net_julia = pd.DataFrame(columns=columns, index=columns, dtype=float).fillna(0)

    for row in nn.itertuples():
        if row[3] > 0:
            net_julia.loc[row[1], row[2]] = net_julia.loc[row[2], row[1]] = row[3]

    if fill_diagonal:
        np.fill_diagonal(net_julia.values, net_julia.sum(axis=1).values)
    return net_julia


def flatten(lst):
    """Flatten a list."""
    return [y for l in lst for y in flatten(l)] if isinstance(lst, (list, np.ndarray)) else [lst]


def upper_to_full(a):
    """Convert the upper part to a full symmetric matrix."""
    n = int((np.sqrt(1 + 8 * a.shape[0]) - 1) / 2)
    A = np.zeros((n, n))
    idx = np.triu_indices(n)
    A[idx] = A[idx[::-1]] = a
    return A


def compose(*functions):
    """Compose two or more functions."""

    def compose2(f, g):
        return lambda x: f(g(x))

    return functools.reduce(compose2, functions, lambda x: x)


def convert_data_to_2d(data):
    """Utility to help move to the new API.

    Data are 3 dimensional with the first dimension representing classes or
    time. The first dimension is compressed, and the belongin of the samples
    to the class is encoded in y.
    """
    X = np.vstack(data)
    y = np.array([np.ones(x.shape[0]) * i for i, x in enumerate(data)]).flatten().astype(int)
    return X, y


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


def normalize_matrix(x):
    """Normalize a matrix so to have 1 on the diagonal, in-place."""
    d = np.diag(x).reshape(1, x.shape[0])
    d = 1.0 / np.sqrt(d)
    x *= d
    x *= d.T


def error_norm(
    cov, comp_cov, norm="frobenius", scaling=True, squared=True, upper_triangular=False, nonzero=False, n=False
):
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
    if n:
        comp_cov = comp_cov.copy()
        # / comp_cov.max()
        cov = cov.copy()
        # / cov.max()
        [normalize_matrix(c) for c in comp_cov]
        [normalize_matrix(c) for c in cov]

    # compute the error
    if upper_triangular:
        comp_cov = np.triu(comp_cov, 1)
        cov = np.triu(cov, 1)
    if nonzero:
        comp_cov = comp_cov[np.where(cov == 0)]
        cov = cov[np.where(cov == 0)]

    error = comp_cov - cov
    # compute the error norm
    if norm == "frobenius":
        squared_norm = np.sum(error ** 2)
    elif norm == "spectral":
        squared_norm = np.amax(np.linalg.svdvals(np.dot(error.T, error)))
    else:
        raise NotImplementedError("Only spectral and frobenius norms are implemented")
    # optionally scale the error norm
    if scaling:
        scaling_factor = (
            error.shape[0]
            if len(error.shape) < 3
            else (np.prod(error.shape[:2]) * ((error.shape[1] - 1) / 2.0 if upper_triangular else error.shape[1]))
        )
        squared_norm = squared_norm / scaling_factor
    # finally get either the squared norm or the norm
    if squared:
        result = squared_norm
    else:
        result = np.sqrt(squared_norm)
    return result


def error_norm_time(cov, comp_cov, norm="frobenius", scaling=True, squared=True):
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
    return np.mean([error_norm(x, y, norm=norm, scaling=scaling, squared=squared) for x, y in zip(cov, comp_cov)])


def structure_error(true, pred, thresholding=False, eps=1e-2, no_diagonal=False):
    """Error in structure between a precision matrix and predicted.

    Parameters
    ----------
    true: array-like
        True matrix. In grpahical inference, if an entry is different from 0
        it is consider as an edge (inverse covariance).

    pred: array-like, shape=(d,d)
        Predicted matrix. In graphical inference, if an entry is different
        from 0 it is consider as an edge (inverse covariance).

    thresholding: bool, default False,
       Apply a threshold (with eps) to the `pred` matrix.

    eps : float, default = 1e-2
        Apply a threshold (with eps) to the `pred` matrix.

    """
    # avoid inplace modifications
    true = true.copy()
    pred = pred.copy()
    if true.ndim > 2:
        y_true = np.array(flatten([squareform(x, checks=None) for x in true]))
        y_pred = np.array(flatten([squareform(x, checks=None) for x in pred]))
    else:
        y_true = squareform(true, checks=None)
        y_pred = squareform(pred, checks=None)

    average_precision = average_precision_score(y_true > 0, y_pred)
    mcc = matthews_corrcoef(y_true > 0, y_pred > 0)

    if thresholding:
        pred[np.abs(pred) < eps] = 0
    tn_to_remove = 0
    if no_diagonal:
        if true.ndim > 2:
            true = np.array([t - np.diag(np.diag(t)) for t in true])
            pred = np.array([t - np.diag(np.diag(t)) for t in pred])
            tn_to_remove = np.prod(true.shape[:2])
        else:
            true -= np.diag(np.diag(true))
            pred -= np.diag(np.diag(pred))
            tn_to_remove = true.shape[0]
    true[true != 0] = 1
    pred[pred != 0] = 2
    res = true + pred
    # from collections import Counter
    # c = Counter(res.flat)
    # tn, fn, fp, tp = c[0], c[1], c[2], c[3]
    TN = np.count_nonzero((res == 0).astype(float)) - tn_to_remove
    FN = np.count_nonzero((res == 1).astype(float))
    FP = np.count_nonzero((res == 2).astype(float))
    TP = np.count_nonzero((res == 3).astype(float))

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN)
    miss_rate = FN / (TP + FN) or 1 - recall
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    accuracy = (TP + TN) / true.size
    prevalence = (TP + FN) / true.size

    fall_out = FP / (FP + TN) if (FP + TN) > 0 else 1
    specificity = TN / (FP + TN) if (FP + TN) > 0 else 1.0 - fall_out

    balanced_accuracy = 0.5 * (recall + specificity)
    false_discovery_rate = FP / (TP + FP) if TP + FP > 0 else 1 - precision
    false_omission_rate = FN / (FN + TN) if FN + TN > 0 else 0
    negative_predicted_value = TN / (FN + TN) if FN + TN > 0 else 1 - false_omission_rate

    positive_likelihood_ratio = recall / fall_out if fall_out > 0 else 0
    negative_likelihood_ratio = miss_rate / specificity if specificity > 0 else 0
    diagnostic_odds_ratio = (
        positive_likelihood_ratio / negative_likelihood_ratio if negative_likelihood_ratio > 0 else 0
    )

    dictionary = dict(
        tp=TP,
        tn=TN,
        fp=FP,
        fn=FN,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        false_omission_rate=false_omission_rate,
        fdr=false_discovery_rate,
        npv=negative_predicted_value,
        prevalence=prevalence,
        miss_rate=miss_rate,
        fall_out=fall_out,
        specificity=specificity,
        plr=positive_likelihood_ratio,
        nlr=negative_likelihood_ratio,
        dor=diagnostic_odds_ratio,
        balanced_accuracy=balanced_accuracy,
        average_precision=average_precision,
        mcc=mcc,
    )
    return dictionary


def mean_structure_error(true, preds):
    """
    Mean and std error in structure between a precision matrix and more
    predicted matrices.

    Parameters
    ----------
    true: array-like
        True matrix. In grpahical inference, if an entry is different from 0
        it is consider as an edge (inverse covariance).

    preds: list of arrays, shape=k*(d,d)
        Predicted matrices. In graphical inference, if an entry is different
        from 0 it is consider as an edge (inverse covariance).
    """
    dictionary = dict(
        tp=[],
        tn=[],
        fp=[],
        fn=[],
        precision=[],
        recall=[],
        f1=[],
        accuracy=[],
        false_omission_rate=[],
        fdr=[],
        npv=[],
        prevalence=[],
        miss_rate=[],
        fall_out=[],
        specificity=[],
        plr=[],
        nlr=[],
        dor=[],
        balanced_accuracy=[],
        average_precision=[],
    )
    for p in preds:
        res = structure_error(true, p, no_diagonal=True)
        for k, v in res.items():
            dictionary[k].append(v)
    res = {}
    for k, l in dictionary.items():
        res[k] = str(np.mean(l)) + "+/-" + str(np.std(l))
    return res


def is_pos_semidef(x, tol=1e-15):
    """Check if x is positive semi-definite."""
    eigs = np.linalg.eigvalsh(x)
    eigs[np.abs(eigs) < tol] = 0
    return np.all(eigs >= 0)


def is_pos_def(x, tol=1e-15, chol=True):
    """Check if x is positive definite."""
    if chol:
        try:
            np.linalg.cholesky(x)
            return True
        except LinAlgError:
            return False

    eigs = np.linalg.eigvalsh(x)
    eigs[np.abs(eigs) < tol] = 0
    return np.all(eigs > 0)


def positive_definite(x, tol=1e-15):
    if x.ndim == 2:
        return is_pos_def(x)
    return all(is_pos_def(y) for y in x)


def ensure_posdef(X, inplace=True):
    def _ensure_posdef_2d(X, inplace=True):
        if not inplace:
            raise NotImplementedError("Only inplace implemented")
        if positive_definite(X):
            return
        X.flat[:: X.shape[0] + 1] = np.abs(X - np.diag(np.diag(X))).sum(axis=1) + 0.1

    if X.ndim == 2:
        return _ensure_posdef_2d(X, inplace)
    for x in X:
        _ensure_posdef_2d(x, inplace)
    return


def threshold(a, threshmin=None, threshmax=None, newval=0):
    """Threshold. Replaces the deprecated scipy function."""
    a = np.ma.array(a, copy=True)
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin is not None:
        mask |= (a < threshmin).filled(False)

    if threshmax is not None:
        mask |= (a > threshmax).filled(False)

    a[mask] = newval
    return a


def alpha_heuristic(emp_cov, n_samples, gamma=0.1):
    """An heuristic for GraphLasso alpha.

    XXX - need testing

    References
    ----------
    http://people.eecs.berkeley.edu/~elghaoui/Pubs/CvxTechCovSel_ICML.pdf
    """
    if emp_cov.ndim == 3:
        diag = np.diagonal(emp_cov, axis1=1, axis2=2)
        m = np.array([d[:, None].dot(d[None, :]) for d in diag]).max()
    elif emp_cov.ndim == 2:
        diag = np.diag(emp_cov)[:, None]
        m = np.max(diag.dot(diag.T))
    else:
        raise ValueError(emp_cov.ndim)

    t = stats.t.pdf(gamma, n_samples - 2) * 2
    num = t * m
    den = np.sqrt(n_samples - 2 + t * t)
    return num / den
