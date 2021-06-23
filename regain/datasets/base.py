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
"""Dataset generation module."""
from __future__ import division

from functools import partial

import numpy as np
from sklearn.utils import Bunch

from regain.datasets.gaussian import data_Meinshausen_Yuan
from regain.datasets.gaussian import data_Meinshausen_Yuan_sparse_latent
from regain.datasets.gaussian import make_covariance
from regain.datasets.gaussian import make_fede
from regain.datasets.gaussian import make_fixed_sparsity
from regain.datasets.gaussian import make_ma_xue_zou
from regain.datasets.gaussian import make_ma_xue_zou_rand_k
from regain.datasets.gaussian import make_sin
from regain.datasets.gaussian import make_sin_cos
from regain.datasets.gaussian import make_sparse_low_rank
from regain.datasets.ising import ising_sampler
from regain.datasets.ising import ising_theta_generator
from regain.datasets.kernels import make_exp_sine_squared
from regain.datasets.kernels import make_ticc
from regain.datasets.poisson import poisson_sampler
from regain.datasets.poisson import poisson_theta_generator


def _gaussian_case(
    n_samples=100,
    n_dim_obs=100,
    n_dim_lat=10,
    T=10,
    mode=None,
    time_on_axis="first",
    update_ell="l2",
    update_theta="l2",
    normalize_starting_matrices=False,
    degree=2,
    epsilon=1e-2,
    keep_sparsity=False,
    proportional=False,
    **kwargs
):
    modes = dict(
        my=data_Meinshausen_Yuan,
        mys=data_Meinshausen_Yuan_sparse_latent,
        sin=make_sin,
        fixed_sparsity=make_fixed_sparsity,
        sincos=make_sin_cos,
        gp=make_exp_sine_squared,
        fede=make_fede,
        sklearn=make_sparse_low_rank,
        ma=make_ma_xue_zou,
        mak=make_ma_xue_zou_rand_k,
        ticc=make_ticc,
    )

    if mode is not None:
        # mode overrides other parameters, for back compatibility
        func = mode if callable(mode) else modes.get(mode, None)
        if func is None:
            raise ValueError("Unknown mode %s. " "Choices are: %s" % (mode, list(modes.keys())))
        kwargs.update(degree=degree, epsilon=epsilon, keep_sparsity=keep_sparsity, proportional=proportional)
    else:
        func = partial(
            make_covariance,
            update_ell=update_ell,
            update_theta=update_theta,
            normalize_starting_matrices=normalize_starting_matrices,
            degree=degree,
            epsilon=epsilon,
            keep_sparsity=keep_sparsity,
            proportional=proportional,
        )
    #     func = partial(
    #         _gaussian_case, mode='ma',
    #         update_ell=update_ell, update_theta=update_theta,
    #         normalize_starting_matrices=normalize_starting_matrices,
    #         degree=degree, epsilon=epsilon, keep_sparsity=keep_sparsity,
    #         proportional=proportional)
    thetas, thetas_obs, ells = func(n_dim_obs=n_dim_obs, n_dim_lat=n_dim_lat, T=T, **kwargs)
    sigmas = list(map(np.linalg.inv, thetas_obs))
    # map(normalize_matrix, sigmas)  # in place

    data = np.array([np.random.multivariate_normal(np.zeros(n_dim_obs), sigma, size=n_samples) for sigma in sigmas])

    X = np.vstack(data)
    y = np.repeat(range(len(sigmas)), n_samples).astype(int)

    if time_on_axis == "last":
        data = data.transpose(1, 2, 0)
    return Bunch(
        data=data, thetas=np.array(thetas), X=X, y=y, thetas_observed=np.array(thetas_obs), ells=np.array(ells)
    )


def _ising_case(
    n_samples=100, n_dim_obs=100, T=10, time_on_axis="first", update_theta="l2", responses=[-1, 1], **kwargs
):
    thetas = ising_theta_generator(n_dim_obs=n_dim_obs, n=n_samples, T=T, mode=update_theta, **kwargs)
    samples = [ising_sampler(t, np.zeros(n_dim_obs), n=n_samples, responses=[-1, 1]) for t in thetas]
    data = np.array(samples)
    X = np.vstack(data)
    y = np.repeat(range(len(thetas)), n_samples).astype(int)
    if time_on_axis == "last":
        data = data.transpose(1, 2, 0)
    return Bunch(data=data, thetas=np.array(thetas), X=X, y=y)


def _poisson_case(n_samples=100, n_dim_obs=100, T=10, time_on_axis="first", update_theta="l1", **kwargs):
    thetas = poisson_theta_generator(n_dim_obs=n_dim_obs, T=T, mode=update_theta, **kwargs)
    samples = [poisson_sampler(t, variances=np.zeros(n_dim_obs), n_samples=n_samples) for t in thetas]
    data = np.array(samples)
    X = np.vstack(data)
    y = np.repeat(range(len(thetas)), n_samples).astype(int)
    if time_on_axis == "last":
        data = data.transpose(1, 2, 0)
    return Bunch(data=data, thetas=np.array(thetas), X=X, y=y)


def make_dataset(
    n_samples=100,
    n_dim_obs=100,
    n_dim_lat=10,
    T=10,
    mode=None,
    time_on_axis="first",
    update_ell="l2",
    update_theta="l2",
    normalize_starting_matrices=False,
    degree=2,
    epsilon=1e-2,
    keep_sparsity=False,
    proportional=False,
    distribution="gaussian",
    **kwargs
):
    """Generate a synthetic dataset.

    Parameters
    ----------
    n_samples: int,
        number of samples to generate
    n_dim_obs: int,
        number of observed variables of the graph
    n_dim_lat: int,
        number of latent variables of the graph
    T: int,
        number of times
    mode: string,   # TO ADD ALL THE POSSIBLE MODES
        "evolving": generate a dataset with evolving observed and latent
                    variables that have a small Frobenious norm between two
                    close time points
        "fixed": generate a dataset with evolving observed and fixed latent
                    variables that have a small Frobenious norm between two
                    close time points
        "l1": generate a dataset with evolving observed and latent variables
              that differs for a small l1 norm
        "l1l2": generate a dataset with evolving observed variables that
                differs for a small l1 norm and evolving latent variables
                that differs for a small l2 norm
        "sin": generate a dataset with fixed latent variables and evolving
                observed variables that are generated from sin functions.
        func : use the user-defined function to generate the dataset. Such
            function should return, in this order, "thetas, thetas_obs, ells".
            See the other functions for an example.
    distribution: string, default='gaussian'
        The distribution considered for the generation of data.
        Options are 'gaussian', 'ising', 'poisson'.
    *kwargs: other arguments related to each specific data generation mode

    """
    n_dim_obs = int(n_dim_obs)
    n_dim_lat = int(n_dim_lat)
    n_samples = int(n_samples)

    if distribution.lower() == "gaussian":
        return _gaussian_case(
            n_samples=n_samples,
            n_dim_obs=n_dim_obs,
            n_dim_lat=n_dim_lat,
            T=T,
            mode=mode,
            time_on_axis=time_on_axis,
            update_ell=update_ell,
            update_theta=update_theta,
            normalize_starting_matrices=normalize_starting_matrices,
            degree=degree,
            epsilon=epsilon,
            keep_sparsity=keep_sparsity,
            proportional=proportional,
            **kwargs
        )

    elif distribution.lower() == "ising":
        return _ising_case(
            n_samples=n_samples,
            n_dim_obs=n_dim_obs,
            T=T,
            time_on_axis=time_on_axis,
            update_theta=update_theta,
            responses=[-1, 1],
            **kwargs
        )
    elif distribution.lower() == "poisson":
        return _poisson_case(
            n_samples=n_samples,
            n_dim_obs=n_dim_obs,
            T=T,
            time_on_axis=time_on_axis,
            update_theta=update_theta,
            **kwargs
        )
    else:
        raise ValueError("distribution `%s` undefined" % distribution)
