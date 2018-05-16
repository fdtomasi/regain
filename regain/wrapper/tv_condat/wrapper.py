import matlab
import matlab.engine
import numpy as np
import os

matlab_engine = matlab.engine.start_matlab()


def total_variation_condat(y, lamda, verbose=False):
    # if matlab_engine is None or not matlab_engine._check_matlab():
    #     if verbose:
    #         print("Starting matlab engine ...")
    #     close_engine = True
    #     matlab_engine = matlab.engine.start_matlab()
    # else:
    #     close_engine = False
    global matlab_engine

    matlab_engine.addpath(
        os.path.abspath(os.path.dirname(__file__)), nargout=0)

    if verbose:
        print("Start GLOPRIDU algorithm ...")
    x = matlab_engine.TV_Condat_v2(
        matlab.double(y[:, None].tolist()), float(lamda))

    # if close_engine:
    #     matlab_engine.quit()
    x = np.asarray(x).ravel()
    return x
