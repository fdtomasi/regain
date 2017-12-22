"""Validation module for REGAIN."""
import numpy as np

from functools import partial
from sklearn.utils.extmath import squared_norm

from regain.norm import l1_norm, node_penalty
from regain.prox import soft_thresholding_sign
from regain.prox import blockwise_soft_thresholding, prox_linf
from regain.prox import prox_laplacian, prox_node_penalty


def check_norm_prox(function):
    """Validate function and return norm with associated prox."""
    if function == 'laplacian':
        prox = prox_laplacian
        norm = squared_norm
    elif function == 'l1':
        prox = soft_thresholding_sign
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
        raise ValueError("Value of %s not understood.", function)
    return norm, prox, function == 'node'
