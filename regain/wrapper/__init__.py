# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
try:
    import matlab
    import matlab.engine
except ImportError:
    # raise ImportError(
    #     "`matlab` package not found. "
    #     "Please note you will need Matlab >= 2016b to use it.")
    pass

import numpy as np
import os

matlab_engine = None  # matlab.engine.start_matlab()


def check_matlab_engine(verbose=False):
    global matlab_engine
    if matlab_engine is None or not matlab_engine._check_matlab():
        if verbose:
            print("Starting matlab engine ...")
        # close_engine = True
        try:
            matlab_engine = matlab.engine.start_matlab()
        except NameError as e:
            if "name 'matlab' is not defined" in str(e):
                raise ValueError("`matlab` package not found. " "Please note you will need Matlab >= 2016b to use it.")
    # else:
    #     close_engine = False


def lvglasso(emp_cov, alpha, tau, rho=1, verbose=False, use_octave=True):
    """Wrapper for LVGLASSO in R.

    If emp_cov.ndim > 2, then emp_cov.shape[0] == emp_cov.shape[1].
    In the temporal case, this means that the time is the last dimension.
    """
    lvglasso_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "matlab", "lvglasso")
    if use_octave:
        from oct2py import octave

        octave.addpath(lvglasso_path, nout=0)
        result = octave.LVGLASSO(emp_cov, alpha, tau, rho)
    else:
        global matlab_engine
        check_matlab_engine(verbose=verbose)
        matlab_engine.addpath(lvglasso_path, nargout=0)
        result = matlab_engine.LVGLASSO(matlab.double(emp_cov.tolist()), float(alpha), float(tau), float(rho))
    return result


def total_variation_condat(y, lamda, verbose=False):
    global matlab_engine
    check_matlab_engine(verbose=verbose)

    tv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "tv_condat")
    matlab_engine.addpath(tv_path, nargout=0)

    if verbose:
        print("Start GLOPRIDU algorithm ...")
    x = matlab_engine.TV_Condat_v2(matlab.double(y[:, None].tolist()), float(lamda))

    # if close_engine:
    #     matlab_engine.quit()
    x = np.asarray(x).ravel()
    return x


def group_lasso_overlap_paspal(X, y, groups=(), lamda=0.1, verbose=False, **kwargs):
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

    glopridu_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "GLO_PRIMAL_DUAL_TOOLBOX")
    matlab_engine.addpath(glopridu_path, nargout=0)

    if verbose:
        print("Start GLOPRIDU algorithm ...")
    coef_ = matlab_engine.glopridu_algorithm(
        matlab.double(X.tolist()),
        matlab.double(y[:, None].tolist()),
        [matlab.int32((np.array(x) + 1).tolist()) for x in groups],  # +1 because of the change of indices
        float(lamda),
    )

    # if close_engine:
    #     matlab_engine.quit()
    coef_ = np.asarray(coef_).ravel()
    return coef_, None, np.nan


def test_group_lasso_paspal():
    """Test function for the module."""
    from sklearn.datasets import make_regression

    X, y, coef = make_regression(n_features=10, coef=True, n_informative=5)

    group_lasso_overlap_paspal(X, y, np.ones(10), 0.1)
