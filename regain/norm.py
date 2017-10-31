"""Useful norms."""
import numpy as np


def l2_square_norm(A):
    return np.square(np.linalg.norm(A, 'fro'))


def l1_norm(precision):
    return np.abs(precision).sum()


def l1_od_norm(precision):
    return np.abs(precision).sum() - np.abs(np.diag(precision)).sum()
