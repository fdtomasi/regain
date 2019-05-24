"""Dataset generation module."""
from __future__ import division

import warnings
from functools import partial

import numpy as np
from scipy import signal
from scipy.spatial.distance import squareform
from sklearn.datasets.base import Bunch
from scipy.linalg import pinv
from sklearn.utils import deprecated

from regain.utils import (
    ensure_posdef, is_pos_def, is_pos_semidef, positive_definite)
from regain.generalized_linear_model.sampling import ising_sampler
from regain._datasets.ising import *
from regain._datasets.kernels import *
from regain._datasets.gaussian import *


def _gaussian_case(n_samples=100, n_dim_obs=100, n_dim_lat=10, T=10, mode=None,
                    time_on_axis='first', update_ell='l2', update_theta='l2',
                    normalize_starting_matrices=False, degree=2, epsilon=1e-2,
                    keep_sparsity=False, proportional=False,
                    **kwargs):
    modes = dict(
        #evolving=make_l2l2,
        #fixed=make_l2,
        #fixedl2=make_l2,
        #fixedl1=make_l1,
        #yuan=generate_dataset_yuan,
        #l1l2=generate_dataset_l1l2,
        #norm=make_l2l2_norm,
        l1l1=generate_dataset_l1l1,
        # the previous are deprecated
        my=data_Meinshausen_Yuan,
        mys=data_Meinshausen_Yuan_sparse_latent,
        sin=make_sin,
        fixed_sparsity=make_fixed_sparsity,
        sincos=make_sin_cos,
        gp=make_exp_sine_squared,
        fede=make_fede,
        sklearn=make_sparse_low_rank,
        ma=make_ma_xue_zou,
        mak=make_ma_xue_zou_rand_k)

    if mode is not None:
        # mode overrides other parameters, for back compatibility
        func = mode if callable(mode) else modes.get(mode, None)
        if func is None:
            raise ValueError(
                "Unknown mode %s. "
                "Choices are: %s" % (mode, modes.keys()))
        kwargs.update(
            degree=degree, epsilon=epsilon, keep_sparsity=keep_sparsity,
            proportional=proportional)
    else:
        func = partial(
            make_covariance, update_ell=update_ell, update_theta=update_theta,
            normalize_starting_matrices=normalize_starting_matrices,
            degree=degree, epsilon=epsilon, keep_sparsity=keep_sparsity,
            proportional=proportional)
    thetas, thetas_obs, ells = func(n_dim_obs, n_dim_lat, T, **kwargs)
    sigmas = map(np.linalg.inv, thetas_obs)
    # map(normalize_matrix, sigmas)  # in place

    data = np.array(
        [
            np.random.multivariate_normal(
                np.zeros(n_dim_obs), sigma, size=n_samples) for sigma in sigmas
        ])

    if time_on_axis == "last":
        data = data.transpose(1, 2, 0)
    return Bunch(
        data=data, thetas=np.array(thetas),
        thetas_observed=np.array(thetas_obs), ells=np.array(ells))


def _ising_case(n_samples=100, n_dim_obs=100, T=10,
                        time_on_axis='first',update_theta='l2',
                        responses=[-1,1], **kwargs):
        thetas = ising_theta_generator(p=n_dim_obs, n=n_samples, T=T,
                                        mode=update_theta,
                                        **kwargs)
        samples = [ising_sampler(t, np.zeros(n_dim_obs), n=n_samples,
                    responses=[-1,1]) for t in thetas]
        data = np.array(samples)
        if time_on_axis == "last":
            data = data.transpose(1, 2, 0)
        return data, thetas

def make_dataset(
        n_samples=100, n_dim_obs=100, n_dim_lat=10, T=10, mode=None,
        time_on_axis='first', update_ell='l2', update_theta='l2',
        normalize_starting_matrices=False, degree=2, epsilon=1e-2,
        keep_sparsity=False, proportional=False, distribution='gaussian',
        **kwargs):
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
        Options are 'gaussian' and 'ising'.
    *kwargs: other arguments related to each specific data generation mode

    """
    n_dim_obs = int(n_dim_obs)
    n_dim_lat = int(n_dim_lat)
    n_samples = int(n_samples)

    if distribution.lower() == 'gaussian':
        return _gaussian_case(n_samples=n_samples, n_dim_obs=n_dim_obs,
                              n_dim_lat=n_dim_lat, T=T,
                              mode=mode, time_on_axis=time_on_axis,
                              update_ell=update_ell, update_theta=update_theta,
                              normalize_starting_matrices=normalize_starting_matrices,
                              degree=degree, epsilon=epsilone,
                              keep_sparsity=keep_sparsity,
                              proportional=proportional, **kwargs)

    elif distribution.lower() == 'ising':
        print(update_theta)
        return _ising_case(n_samples=n_samples, n_dim_obs=n_dim_obs,
                                        T=T, time_on_axis=time_on_axis,
                                update_theta=update_theta,
                                responses=[-1,1])
