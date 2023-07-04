from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator


@dataclass
class Convergence:
    iter: float = 0
    obj: float = 0
    iter_norm: float = 0
    iter_r_norm: float = 0

    def __str__(self):
        return f"Iter: {self.iter}, objective: {self.obj:.4f}, iter_norm {self.iter_norm:.4f}, iter_norm_normalized: {self.iter_r_norm:.4f}"


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
