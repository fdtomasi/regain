import matlab
import matlab.engine
import numpy as np
import os

matlab_engine = matlab.engine.start_matlab()


def group_lasso_overlap_paspal(X, y, groups=(), lamda=0.1, verbose=False,
                               **kwargs):
    # if matlab_engine is None or not matlab_engine._check_matlab():
    #     if verbose:
    #         print("Starting matlab engine ...")
    #     close_engine = True
    #     matlab_engine = matlab.engine.start_matlab()
    # else:
    #     close_engine = False
    global matlab_engine

    matlab_engine.addpath(
        os.path.join(os.path.abspath(os.path.dirname(__file__)),
                     'matlab/GLO_PRIMAL_DUAL_TOOLBOX/'), nargout=0)

    if verbose:
        print("Start GLOPRIDU algorithm ...")
    coef_ = matlab_engine.glopridu_algorithm(
        matlab.double(X.tolist()),
        matlab.double(y[:, None].tolist()),
        [matlab.int32((np.array(x) + 1).tolist()) for x in groups],  # +1 because of the change of indices
        float(lamda))

    # if close_engine:
    #     matlab_engine.quit()
    coef_ = np.asarray(coef_).ravel()
    return coef_, None, np.nan


def test():
    from sklearn.datasets import make_regression
    X, y, coef = make_regression(n_features=10, coef=True, n_informative=5)

    group_lasso_overlap_paspal(X, y, np.ones(10), 0.1)
