from __future__ import division, print_function

import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.gaussian_process import kernels
from sklearn.model_selection import KFold, StratifiedKFold

from regain.covariance import latent_time_graphical_lasso_, time_graphical_lasso_
from regain import datasets, utils
from regain.bayesian import wishart_process_
from regain.covariance import (
    kernel_latent_time_graphical_lasso_, kernel_time_graphical_lasso_)
from skopt.searchcv import BayesSearchCV


def use_bscv(mdl, search_spaces, data, y=None):
    # n_iter = 100 if isinstance(
    #     mdl, (
    #         kernel_time_graphical_lasso_.KernelTimeGraphicalLasso,
    #         kernel_latent_time_graphical_lasso_.KernelLatentTimeGraphicalLasso
    #     )) else 50
    # n_iter = 10**len(search_spaces.keys())
    n_iter = 100
    bscv = BayesSearchCV(
        mdl, search_spaces=search_spaces, n_iter=n_iter, n_points=3, n_jobs=-1,
        verbose=False, cv=StratifiedKFold(3)
        if y is not None else KFold(3), error_score=-np.inf)
    bscv.fit(data, y)
    return bscv.best_estimator_


def base_results(mdl, X, y, K, K_obs, ells, search_spaces=None, **params):
    ll = mdl.set_params(**params)

    tic = time.time()
    if search_spaces is None:
        ll.fit(X, y)
    else:
        ll = use_bscv(ll, search_spaces, X, y)
    tac = time.time()

    ss = utils.structure_error(K, ll.precision_)
    MSE_precision = utils.error_norm(K, ll.precision_, upper_triangular=True)

    res = dict(
        n_dim_obs=K.shape[1], time=tac - tic, iterations=ll.n_iter_,
        MSE_precision=MSE_precision, estimator=ll, likelihood=ll.score(X, y))

    if hasattr(ll, 'latent_'):
        res['MSE_observed'] = utils.error_norm(
            K_obs, ll.precision_ - ll.latent_)
        res['MSE_latent'] = utils.error_norm(ells, ll.latent_)
        res['mean_rank_error'] = utils.error_rank(ells, ll.latent_)

    res = dict(res, **ss)
    return res


def tgl_results(data_grid, K, K_obs, ells, search_spaces=None, **params):
    mdl = time_graphical_lasso_.TimeGraphicalLasso(
        time_on_axis='last', assume_centered=0, verbose=0, rtol=1e-5, tol=1e-5,
        max_iter=1000, rho=1. / np.sqrt(data_grid.shape[0]))

    return base_results(
        mdl, data_grid, None, K, K_obs, ells, search_spaces, **params)


def ltgl_results(data_grid, K, K_obs, ells, search_spaces=None, **params):
    mdl = latent_time_graphical_lasso_.LatentTimeGraphicalLasso(
        time_on_axis='last', assume_centered=0, verbose=0, rtol=1e-5, tol=1e-5,
        max_iter=1000, rho=1. / np.sqrt(data_grid.shape[0]),
        update_rho_options=dict(mu=5))

    return base_results(
        mdl, data_grid, None, K, K_obs, ells, search_spaces, **params)


def ktgl_results(data_list, K, K_obs, ells, search_spaces=None, **params):
    mdl = kernel_time_graphical_lasso_.KernelTimeGraphicalLasso(
        alpha=0.5, psi='laplacian', assume_centered=0, rtol=1e-5, tol=1e-5,
        max_iter=1000, rho=1. / np.sqrt(data_list.shape[1]),
        update_rho_options=dict(mu=5), kernel=partial(
            kernels.ExpSineSquared, periodicity=np.pi), ker_param=2)

    X = np.vstack(data_list)
    y = np.array([np.ones(x.shape[0]) * i
                  for i, x in enumerate(data_list)]).flatten().astype(int)
    return base_results(mdl, X, y, K, K_obs, ells, search_spaces, **params)


def kltgl_results(data_list, K, K_obs, ells, search_spaces=None, **params):
    mdl = kernel_latent_time_graphical_lasso_.KernelLatentTimeGraphicalLasso(
        alpha=0.5,
        rtol=1e-5,
        tol=1e-5,
        max_iter=1000,
        rho=1. / np.sqrt(data_list.shape[1]),
        update_rho_options=dict(mu=5),
        kernel_psi=partial(kernels.ExpSineSquared, periodicity=np.pi),
        ker_psi_param=2,
        kernel_phi=partial(kernels.ExpSineSquared, periodicity=np.pi),
        ker_phi_param=2,
    )

    X = np.vstack(data_list)
    y = np.array([np.ones(x.shape[0]) * i
                  for i, x in enumerate(data_list)]).flatten().astype(int)
    return base_results(mdl, X, y, K, K_obs, ells, search_spaces, **params)


def wp_results(data_list, K, **params):
    n_iter = 1000
    mdl = wishart_process_.WishartProcess(verbose=True, n_iter=n_iter, **params)

    X = np.vstack(data_list)
    y = np.array([np.ones(x.shape[0]) * i
                  for i, x in enumerate(data_list)]).flatten().astype(int)

    tic = time.time()
    ll = mdl.fit(X, y)
    tac = time.time()

    #     mdl.likelihood(wp.D_map)
    #     mdl.loglikes_after_burnin.max()
    mdl.store_precision = True
    ss = utils.structure_error(K, ll.precision_, thresholding=False, eps=1e-3)
    MSE_precision = utils.error_norm(K, ll.precision_, upper_triangular=True)

    res = dict(
        n_dim_obs=K.shape[1], time=tac - tic, iterations=n_iter,
        MSE_precision=MSE_precision, estimator=ll, likelihood=ll.score(X, y))
    res = dict(res, **ss)
    return res


def run_results(data, df, scores):
    idx = pd.IndexSlice
    for i, res in enumerate(data):
        # if i > 3: continue
        # dim = k[0]
        data_list = res.data
        K = res.thetas
        K_obs = res.thetas_observed
        ells = res.ells
        # to use it later for grid search
        data_grid = np.array(data_list).transpose(1, 2, 0)
        T = data_list.shape[0]
        print("Start with: dim=%d, T=%d (it %d)" % (data_list.shape[-1], T, i))

        print("starting TGL ...\r", end='')
        res = tgl_results(
            data_grid,
            K,
            K_obs,
            ells,
            search_spaces={
                'alpha': (1e-4, 1e+1, 'log-uniform'),
                'beta': (1e-4, 1e+1, 'log-uniform'),
            },
        )
        df.loc[idx['TGL', T], idx[:, i]] = [res.get(x, None) for x in scores]

        print("starting KTGL - exp...\r", end='')
        res = ktgl_results(
            data_list, K, K_obs, ells, search_spaces={
                'alpha': (1e-4, 1e+1, 'log-uniform'),
                'ker_param': (1e-4, 1e+1, 'log-uniform')
            })
        df.loc[idx['KTGL-exp', T], idx[:, i]] = [
            res.get(x, None) for x in scores
        ]

        print("starting KTGL - rbf ...\r", end='')
        res = ktgl_results(
            data_list, K, K_obs, ells, search_spaces={
                'alpha': (1e-4, 1e+1, 'log-uniform'),
                'ker_param': (1e-4, 1e+1, 'log-uniform')
            }, kernel=partial(kernels.RBF))
        df.loc[idx['KTGL-rbf', T], idx[:, i]] = [
            res.get(x, None) for x in scores
        ]

        print("starting LTGL ...\r", end='')
        res = ltgl_results(
            data_grid, K, K_obs, ells, search_spaces={
                'alpha': (1e-4, 1e+1, 'log-uniform'),
                'tau': (1e-4, 1e+1, 'log-uniform'),
                'beta': (1e-4, 1e+1, 'log-uniform'),
            }, eta=20)
        df.loc[idx['LTGL', T], idx[:, i]] = [res.get(x, None) for x in scores]
        alpha = res['estimator'].alpha
        tau = res['estimator'].tau

        print("starting KLTGL - exp ...\r", end='')
        res = kltgl_results(
            data_list, K, K_obs, ells, search_spaces={
                # 'alpha': (1e-4, 1e+1, 'log-uniform'),
                # 'tau': (1e-4, 1e+1, 'log-uniform'),
                'ker_psi_param': (1e-4, 1e+1, 'log-uniform'),
            }, alpha=alpha, tau=tau,
            ker_phi_param=20, kernel_phi=partial(kernels.ConstantKernel))
        df.loc[idx['KLTGL-exp', T], idx[:, i]] = [
            res.get(x, None) for x in scores
        ]

        print("starting KLTGL - rbf ...\r", end='')
        res = kltgl_results(
            data_list,
            K,
            K_obs,
            ells,
            search_spaces={
                # 'alpha': (1e-4, 1e+1, 'log-uniform'),
                # 'tau': (1e-4, 1e+1, 'log-uniform'),
                'ker_psi_param': (1e-4, 1e+1, 'log-uniform'),
            },
            alpha=alpha,
            tau=tau,
            ker_phi_param=20,
            kernel_phi=partial(kernels.ConstantKernel),
            kernel_psi=partial(kernels.RBF),
        )
        df.loc[idx['KLTGL-rbf', T], idx[:, i]] = [
            res.get(x, None) for x in scores
        ]

        # print("starting WP ...\r", end='')
        # res = wp_results(data_list, K)
        # df.loc[idx['WP', T], idx[:, i]] = [res.get(x, None) for x in scores]
    return df
