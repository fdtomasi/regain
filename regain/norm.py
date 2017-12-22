"""Useful norms."""
import numpy as np
from scipy.optimize import minimize


def l1_norm(precision):
    """L1 norm."""
    return np.abs(precision).sum()


def l1_od_norm(precision):
    """L1 norm off-diagonal."""
    return np.abs(precision).sum() - np.abs(np.diag(precision)).sum()


def node_penalty(X):
    """Node penalty. See Hallac for details."""
    cons = (
        {'type': 'eq',
         'fun': lambda x: np.array((x.reshape(X.shape) + x.reshape(X.shape).T - X).sum()), 'jac': lambda x: np.full(np.prod(X.shape), 2)},
        # {'type': 'ineq',  # avoid problem with scipy
        #  'fun': lambda x: 1,
        #  'jac': lambda x: np.zeros(np.prod(X.shape))}
    )
    return minimize(
        lambda x: np.sum(np.linalg.norm(x.reshape(X.shape), axis=0)),
        np.random.randn(np.prod(X.shape)), constraints=cons).fun
