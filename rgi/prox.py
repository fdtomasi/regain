import numpy as np
import warnings


def soft_thresholding(a, kappa):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.maximum(0, 1 - kappa / np.linalg.norm(a)) * a
