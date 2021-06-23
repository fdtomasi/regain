import numpy as np

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from regain.utils import namedtuple_with_defaults

convergence = namedtuple_with_defaults("convergence", "iter obj iter_norm iter_r_norm")


def build_adjacency_matrix(neighbours, how="union"):
    out = np.eye(len(neighbours))
    if how.lower() == "union":
        for i, arr in enumerate(neighbours):
            where = [j for j in range(len(neighbours)) if j != i]
            out[i, where] = arr
            out = (out + out.T) / 2
    elif how.lower() == "intersection":
        for i, arr in enumerate(neighbours):
            where = [j for j in range(len(neighbours)) if j != i]
            out[i, where] = arr
        binarized = (out.copy() != 0).astype(int)
        binarized = (binarized + binarized.T) / 2
        binarized[np.where(binarized < 1)] = 0
        out = (out + out.T) / 2
        out[np.where(binarized == 0)] = 0
        assert np.all(out == out.T)
    return out


class GLM_GM(ABC, BaseEstimator):
    def __init__(
        self,
        alpha=0.01,
        tol=1e-4,
        rtol=1e-4,
        max_iter=100,
        verbose=False,
        return_history=True,
        return_n_iter=False,
        compute_objective=True,
    ):
        self.alpha = alpha
        self.tol = tol
        self.rtol = rtol
        self.max_iter = max_iter
        self.verbose = verbose
        self.return_history = return_history
        self.return_n_iter = return_n_iter
        self.compute_objective = compute_objective

    @abstractmethod
    def fit(self, X, y=None, gamma=1e-3):
        pass
