try:
    import matlab
    import matlab.engine
except ImportError:
    raise ImportError(
        "`matlab` package not found. "
        "Please note you will need Matlab >= 2016b to use it.")

import numpy as np
import os


matlab_engine = None  # matlab.engine.start_matlab()


def check_matlab_engine(verbose=False):
    global matlab_engine
    if matlab_engine is None or not matlab_engine._check_matlab():
        if verbose:
            print("Starting matlab engine ...")
        # close_engine = True
        matlab_engine = matlab.engine.start_matlab()
    # else:
    #     close_engine = False


def lvglasso(emp_cov, alpha, tau, rho=1, verbose=False):
    global matlab_engine
    check_matlab_engine(verbose=verbose)

    lvglasso_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'lvglasso_')
    matlab_engine.addpath(lvglasso_path, nargout=0)

    if emp_cov.ndim > 2:
        result = matlab_engine.LVGLASSO(
            matlab.double(emp_cov.tolist()), float(alpha), float(tau), float(rho))
    else:
        result = matlab_engine.LVGLASSO_single_time(
            matlab.double(emp_cov.tolist()), float(alpha), float(tau), float(rho))
    return result


def total_variation_condat(y, lamda, verbose=False):
    global matlab_engine
    check_matlab_engine(verbose=verbose)

    tv_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'tv_condat')
    matlab_engine.addpath(tv_path, nargout=0)

    if verbose:
        print("Start GLOPRIDU algorithm ...")
    x = matlab_engine.TV_Condat_v2(
        matlab.double(y[:, None].tolist()), float(lamda))

    # if close_engine:
    #     matlab_engine.quit()
    x = np.asarray(x).ravel()
    return x


def group_lasso_overlap_paspal(
        X, y, groups=(), lamda=0.1, verbose=False, **kwargs):
    """Group Lasso with Overlap via PASPAL (Matlab implementation).

    Parameters
    ----------
    X : ndarray
        Data.
    y : ndarray
        Classes.
    groups : list-type
        Groups of variables.
    lamda : float
        Regularization parameter.
    verbose : boolean
        If True, print debug information.

    Returns
    -------
    coef_
        Coefficient of the Lasso algorithm for each feature.

    """
    global matlab_engine
    check_matlab_engine(verbose=verbose)

    glopridu_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'GLO_PRIMAL_DUAL_TOOLBOX')
    matlab_engine.addpath(glopridu_path, nargout=0)

    if verbose:
        print("Start GLOPRIDU algorithm ...")
    coef_ = matlab_engine.glopridu_algorithm(
        matlab.double(X.tolist()),
        matlab.double(y[:, None].tolist()),
        [matlab.int32((np.array(x) + 1).tolist())
         for x in groups],  # +1 because of the change of indices
        float(lamda))

    # if close_engine:
    #     matlab_engine.quit()
    coef_ = np.asarray(coef_).ravel()
    return coef_, None, np.nan


def test_group_lasso_paspal():
    """Test function for the module."""
    from sklearn.datasets import make_regression
    X, y, coef = make_regression(n_features=10, coef=True, n_informative=5)

    group_lasso_overlap_paspal(X, y, np.ones(10), 0.1)
