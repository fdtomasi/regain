"""Validation module for REGAIN."""
import warnings
from functools import partial

import numpy as np
from sklearn.utils.extmath import squared_norm

from regain.norm import l1_norm, node_penalty
from regain.prox import (blockwise_soft_thresholding, prox_laplacian,
                         prox_linf, prox_node_penalty, soft_thresholding)


def check_norm_prox(function):
    """Validate function and return norm with associated prox."""
    if function == 'laplacian':
        prox = prox_laplacian
        norm = squared_norm
    elif function == 'l1':
        prox = soft_thresholding
        norm = l1_norm
    elif function == 'l2':
        prox = blockwise_soft_thresholding
        norm = np.linalg.norm
    elif function == 'linf':
        prox = prox_linf
        norm = partial(np.linalg.norm, ord=np.inf)
    elif function == 'node':
        prox = prox_node_penalty
        norm = node_penalty
    else:
        raise ValueError("Value of %s not understood." % function)
    return norm, prox, function == 'node'


def check_array_dimensions(X, n_dimensions=3, time_on_axis='first'):
    """Validate input matrix."""
    if isinstance(X, list):
        warnings.warn("Input data is list; assumed to be a list of matrices "
                      "for each time point")
        return X
    if X.ndim == n_dimensions - 1:
        warnings.warn("Input data should have %d"
                      " dimensions, found %d. Reshaping input into %s"
                      % (n_dimensions, X.ndim, (1,) + X.shape))
        return X[None, ...]

    elif X.ndim != n_dimensions:
        raise ValueError("Input data should have %d"
                         " dimensions, found %d." % (n_dimensions, X.ndim))

    if time_on_axis == 'last':
        return X.transpose(2, 0, 1)  # put time as first dimension

    return X
