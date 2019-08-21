try:
    import prox_tv
except ImportError:
    import warnings
    warnings.warn(
        "Forward-backward minimisation method relies on `prox_tv` "
        "library. Please install it before using this class.")