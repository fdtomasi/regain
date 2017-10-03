import numpy as np


def flatten(lst):
    """Flatten a list."""
    return [y for l in lst for y in flatten(l)] \
        if isinstance(lst, (list, np.ndarray)) else [lst]
