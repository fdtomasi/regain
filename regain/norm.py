"""Useful norms."""
import numpy as np
from scipy.optimize import minimize


def vector_p_norm(a, p=1):
    """Sum of norms for each vector."""
    b = np.array([b.flatten() for b in a]).T
    return np.linalg.norm(b, axis=1, ord=1).sum()


def l1_norm(precision):
    """L1 norm."""
    return np.abs(precision).sum()


def l1_od_norm(precision):
    """L1 norm off-diagonal."""
    return l1_norm(precision) - np.abs(np.diag(precision)).sum()


def node_penalty(X):
    """Node penalty. See Hallac for details."""
    cons = (
        {'type': 'eq',
         'fun': lambda x: np.array((x.reshape(X.shape) + x.reshape(X.shape).T - X).sum()), 'jac': lambda x: np.full(X.size, 2)},
    )
    try:
        res = minimize(
            lambda x: np.sum(np.linalg.norm(x.reshape(X.shape), axis=0)),
            np.random.randn(X.size), constraints=cons).fun
    except ValueError:
        res = np.nan

    return res
