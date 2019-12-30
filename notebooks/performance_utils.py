from regain.wrapper import lvglasso
from sklearn.datasets.base import Bunch
import numpy as np
import time
from sklearn.covariance import log_likelihood, empirical_covariance, GraphicalLasso as GLsk

from regain.covariance import GraphicalLasso, LatentGraphicalLasso, TimeGraphicalLasso, LatentTimeGraphicalLasso
from regain import utils


def likelihood_score(X, precision_):
    # compute empirical covariance of the test set
    location_ = X.mean(1).reshape(X.shape[0], 1, X.shape[2])
    test_cov = np.array(
        [empirical_covariance(x, assume_centered=True) for x in X - location_])

    res = sum(log_likelihood(S, K) for S, K in zip(test_cov, precision_))

    return res


def gl_results(data_grid, K, K_obs, ells, **params):
    mdl = GraphicalLasso(
        assume_centered=0, verbose=0, rtol=1e-5, tol=1e-5, max_iter=500,
        rho=1. / np.sqrt(data_grid.shape[0]))

    tic = time.time()
    iters = []
    precisions = []
    for d in data_grid.transpose(2, 0, 1):
        mdl.set_params(**params).fit(d)
        iters.append(mdl.n_iter_)
        precisions.append(mdl.precision_)
    tac = time.time()
    iterations = np.max(iters)
    precisions = np.array(precisions)

    F1score = utils.structure_error(K, precisions)['f1']
    MSE_observed = None
    MSE_precision = utils.error_norm(K, precisions)
    MSE_latent = None
    mean_rank_error = None

    res = dict(
        n_dim_obs=K.shape[1], time=tac - tic, iterations=iterations,
        F1score=F1score, MSE_precision=MSE_precision,
        MSE_observed=MSE_observed, MSE_latent=MSE_latent,
        mean_rank_error=mean_rank_error, note=None, estimator=mdl)
    return res


def lgl_results(data_grid, K, K_obs, ells, **params):
    mdl = LatentGraphicalLasso(
        assume_centered=0, verbose=0, rtol=1e-5, tol=1e-5, max_iter=500,
        rho=1. / np.sqrt(data_grid.shape[0]))

    tic = time.time()
    iters = []
    precisions, latents = [], []
    for d in data_grid.transpose(2, 0, 1):
        mdl.set_params(**params).fit(d)
        iters.append(mdl.n_iter_)
        precisions.append(mdl.precision_)
        latents.append(mdl.latent_)
    tac = time.time()
    iterations = np.max(iters)
    precisions = np.array(precisions)
    latents = np.array(latents)

    F1score = utils.structure_error(K, precisions)['f1']
    MSE_observed = utils.error_norm(K_obs, precisions - latents)
    MSE_precision = utils.error_norm(K, precisions)
    MSE_latent = utils.error_norm(ells, latents)
    mean_rank_error = utils.error_rank(ells, latents)

    res = dict(
        n_dim_obs=K.shape[1], time=tac - tic, iterations=iterations,
        F1score=F1score, MSE_precision=MSE_precision,
        MSE_observed=MSE_observed, MSE_latent=MSE_latent,
        mean_rank_error=mean_rank_error, note=None, estimator=mdl)
    return res


def tgl_results(X, y, K, K_obs, ells, **params):
    mdl = TimeGraphicalLasso(
        assume_centered=0, verbose=0, rtol=1e-5, tol=1e-5,
        max_iter=500, rho=1. / np.sqrt(X.shape[0]))

    tic = time.time()
    ll = mdl.set_params(**params).fit(X, y)
    tac = time.time()
    iterations = ll.n_iter_
    F1score = utils.structure_error(K, ll.precision_)['f1']
    MSE_observed = None  # utils.error_norm(K_obs, ll.precision_ - ll.latent_)
    MSE_precision = utils.error_norm(K, ll.precision_)
    MSE_latent = None  # utils.error_norm(ells, ll.latent_)
    mean_rank_error = None  # utils.error_rank(ells, ll.latent_)

    res = dict(
        n_dim_obs=K.shape[1], time=tac - tic, iterations=iterations,
        F1score=F1score, MSE_precision=MSE_precision,
        MSE_observed=MSE_observed, MSE_latent=MSE_latent,
        mean_rank_error=mean_rank_error, likelihood=mdl.score(X, y), note=None,
        estimator=ll)
    return res


def ltgl_results(X, y, K, K_obs, ells, **params):
    mdl = LatentTimeGraphicalLasso(
        assume_centered=0, verbose=0, rtol=1e-5, tol=1e-5,
        max_iter=1000, rho=1. / np.sqrt(X.shape[0]),
        update_rho_options=dict(mu=5))

    tic = time.time()
    ll = mdl.set_params(**params).fit(X, y)
    tac = time.time()
    iterations = ll.n_iter_
    ss = utils.structure_error(K, ll.precision_)  #, thresholding=1, eps=1e-5)
    MSE_observed = utils.error_norm(K_obs, ll.precision_ - ll.latent_)
    MSE_precision = utils.error_norm(K, ll.precision_, upper_triangular=True)
    MSE_latent = utils.error_norm(ells, ll.latent_)
    mean_rank_error = utils.error_rank(ells, ll.latent_)

    res = dict(
        n_dim_obs=K.shape[1], time=tac - tic, iterations=iterations,
        MSE_precision=MSE_precision, MSE_observed=MSE_observed,
        MSE_latent=MSE_latent, mean_rank_error=mean_rank_error, note=None,
        estimator=ll, likelihood=mdl.score(X, y), latent=ll.latent_)

    res = dict(res, **ss)
    return res


def glasso_results(data_grid, K, K_obs, ells, alpha):
    gl = GLsk(alpha=alpha, mode='cd', assume_centered=False, max_iter=500)

    tic = time.time()
    iters = []
    precisions = []
    for d in data_grid.transpose(2, 0, 1):
        gl.fit(d)
        iters.append(gl.n_iter_)
        precisions.append(gl.precision_)
    tac = time.time()
    iterations = np.max(iters)
    precisions = np.array(precisions)

    ss = utils.structure_error(K, precisions)  #, thresholding=1, eps=1e-5)

    MSE_observed = None
    MSE_precision = utils.error_norm(K, precisions, upper_triangular=True)
    MSE_latent = None
    mean_rank_error = None

    res = dict(
        n_dim_obs=K.shape[1], time=tac - tic, iterations=iterations,
        MSE_precision=MSE_precision, MSE_observed=MSE_observed,
        MSE_latent=MSE_latent, mean_rank_error=mean_rank_error,
        likelihood=likelihood_score(data_grid.transpose(2, 0, 1),
                                    precisions), note=None, estimator=gl)

    res = dict(res, **ss)
    return res


def friedman_results(data_grid, K, K_obs, ells, alpha):
    from rpy2.robjects.packages import importr
    glasso = importr('glasso').glasso

    tic = time.time()
    iters = []
    precisions = []
    for d in data_grid.transpose(2, 0, 1):
        emp_cov = empirical_covariance(d)
        out = glasso(emp_cov, alpha)
        iters.append(int(out[-1][0]))
        precisions.append(np.array(out[1]))
    tac = time.time()
    iterations = np.max(iters)
    precisions = np.array(precisions)
    F1score = utils.structure_error(K, precisions)['f1']
    MSE_observed = None
    MSE_precision = utils.error_norm(K, precisions, upper_triangular=True)
    MSE_latent = None
    mean_rank_error = None

    res = dict(
        n_dim_obs=K.shape[1], time=tac - tic, iterations=iterations,
        F1score=F1score, MSE_precision=MSE_precision,
        MSE_observed=MSE_observed, MSE_latent=MSE_latent,
        mean_rank_error=mean_rank_error,
        likelihood=likelihood_score(data_grid.transpose(2, 0, 1),
                                    precisions), note=None, estimator=None)

    return res


def hallac_results(
        data_grid, K, K_obs, ells, beta, alpha, penalty=2, tvgl_path=''):
    if tvgl_path:
        import sys
        sys.path.append(tvgl_path)
        import TVGL
    #     with suppress_stdout():
    tic = time.time()
    thetaSet, empCovSet, status, gvx = TVGL.TVGL(
        np.vstack(data_grid.transpose(2, 0, 1)), data_grid.shape[0],
        lamb=alpha, beta=beta, indexOfPenalty=penalty)
    tac = time.time()

    if status != "Optimal":
        print("not converged")
    precisions = np.array(thetaSet)
    ss = utils.structure_error(K, precisions)
    MSE_observed = None
    MSE_precision = utils.error_norm(K, precisions, upper_triangular=True)
    MSE_latent = None
    mean_rank_error = None

    res = dict(
        n_dim_obs=K.shape[1], time=tac - tic, iterations=gvx.n_iter_,
        MSE_precision=MSE_precision, MSE_observed=MSE_observed,
        MSE_latent=MSE_latent, mean_rank_error=mean_rank_error,
        likelihood=likelihood_score(data_grid.transpose(2, 0, 1),
                                    precisions), note=status, estimator=gvx)
    res = dict(res, **ss)

    return res


def chandresekeran_results(data_grid, K, K_obs, ells, tau, alpha, **kwargs):
    emp_cov = np.array(
        [
            empirical_covariance(x, assume_centered=True)
            for x in data_grid.transpose(2, 0, 1)
        ]).transpose(1, 2, 0)

    rho = 1. / np.sqrt(data_grid.shape[0])

    result = lvglasso(emp_cov, alpha, tau, rho)
    ma_output = Bunch(**result)

    R = np.array(ma_output.R).T
    S = np.array(ma_output.S).T
    L = np.array(ma_output.L).T

    ss = utils.structure_error(K, S)
    MSE_observed = utils.error_norm(K_obs, R)
    MSE_precision = utils.error_norm(K, S, upper_triangular=True)
    MSE_latent = utils.error_norm(ells, L)
    mean_rank_error = utils.error_rank(ells, L)

    res = dict(
        n_dim_obs=K.shape[1], time=ma_output.elapsed_time,
        iterations=np.max(ma_output.iter), MSE_precision=MSE_precision,
        MSE_observed=MSE_observed, MSE_latent=MSE_latent,
        mean_rank_error=mean_rank_error, note=None, estimator=ma_output,
        likelihood=likelihood_score(data_grid.transpose(2, 0, 1), R), latent=L)

    res = dict(res, **ss)
    return res
