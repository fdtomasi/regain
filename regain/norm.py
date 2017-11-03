"""Useful norms."""
import numpy as np


def l1_norm(precision):
    return np.abs(precision).sum()


def l1_od_norm(precision):
    return np.abs(precision).sum() - np.abs(np.diag(precision)).sum()
