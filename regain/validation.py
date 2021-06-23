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

"""Validation module for REGAIN."""
import warnings
from functools import partial

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_array

from regain.norm import l1_norm, node_penalty
from regain.prox import blockwise_soft_thresholding, prox_laplacian, prox_linf, prox_node_penalty, soft_thresholding


def check_norm_prox(function):
    """Validate function and return norm with associated prox."""
    if function == "laplacian":
        prox = prox_laplacian
        norm = squared_norm
    elif function == "l1":
        prox = soft_thresholding
        norm = l1_norm
    elif function == "l2":
        prox = blockwise_soft_thresholding
        norm = np.linalg.norm
    elif function == "linf":
        prox = prox_linf
        norm = partial(np.linalg.norm, ord=np.inf)
    elif function == "node":
        prox = prox_node_penalty
        norm = node_penalty
    else:
        raise ValueError("Value of %s not understood." % function)
    return norm, prox, function == "node"


def check_array_dimensions(X, n_dimensions=3, time_on_axis="first", suppress_warn_list=False):
    """Validate input matrix."""
    if isinstance(X, list):
        if not suppress_warn_list:
            warnings.warn("Input data is list; assumed to be a list of matrices " "for each time point")
        return X
    if X.ndim == n_dimensions - 1:
        warnings.warn(
            "Input data should have %d"
            " dimensions, found %d. Reshaping input into %s" % (n_dimensions, X.ndim, (1,) + X.shape)
        )
        return X[None, ...]

    elif X.ndim != n_dimensions:
        raise ValueError("Input data should have %d" " dimensions, found %d." % (n_dimensions, X.ndim))

    if time_on_axis == "last":
        return X.transpose(2, 0, 1)  # put time as first dimension

    return X


def check_input_3d(X, time_on_axis="first", suppress_warn_list=False, estimator=None):
    """Old API. X is a 3d matrix."""
    X = check_array_dimensions(X, n_dimensions=3, time_on_axis=time_on_axis, suppress_warn_list=suppress_warn_list)

    is_list = isinstance(X, list)
    # Covariance does not make sense for a single feature
    X = np.array([check_array(x, ensure_min_features=2, ensure_min_samples=2, estimator=estimator) for x in X])
    if is_list:
        n_times = len(X)
        if np.unique([x.shape[1] for x in X]).size != 1:
            raise ValueError("Input data cannot have different number " "of variables.")
        n_dimensions = X[0].shape[1]
    else:
        n_times = X.shape[0]
        n_dimensions = X.shape[2]

    n_samples = np.array([x.shape[0] for x in X])
    return X, n_samples, n_dimensions, n_times


def check_input(X, y=None, **kwargs):
    """Validate data and labels."""
    if sp.issparse(X):
        raise TypeError("sparse matrices not supported.")

    if y is None:
        # needs X.ndim == 3 or X is list
        return check_input_3d(X, **kwargs)
    else:
        raise ValueError("y cannot be not None with this class.")
