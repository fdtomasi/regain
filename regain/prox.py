import numpy as np
import warnings


def soft_thresholding(a, lamda):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.maximum(0, 1 - lamda / np.linalg.norm(a)) * a


def soft_thresholding_sign(a, lamda):
    return np.sign(a) * np.maximum(np.abs(a) - lamda, 0)


def prox_l1_od(A, lamda):
    Z = np.zeros(A.shape)
    ind = (A > lamda)
    Z[ind] = A[ind] - lamda
    ind = (A < - lamda)
    Z[ind] = A[ind] + lamda
    return Z
