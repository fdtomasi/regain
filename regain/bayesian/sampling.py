import numpy as np
from scipy import linalg, stats
from sklearn.utils.extmath import fast_logdet

from regain.bayesian.wishart_process_ import GWP_construct, log_lik_frob

lognpdf = stats.lognorm.pdf
normpdf = stats.norm.pdf


def elliptical_slice(xx, prior, cur_log_like, sigma2, angle_range=0, *varargin):
    # %ELLIPTICAL_SLICE Markov chain update for a distribution with a Gaussian "prior" factored out
    # %
    # %     [xx, cur_log_like] = elliptical_slice(xx, chol_Sigma, log_like_fn);
    # % OR
    # %     [xx, cur_log_like] = selliptical_slice(xx, prior_sample, log_like_fn);
    # %
    # % Optional additional arguments: cur_log_like, angle_range, varargin (see below).
    # %
    # % A Markov chain update is applied to the D-element array xx leaving a
    # % "posterior" distribution
    # %     P(xx) \propto N(xx;0,Sigma) L(xx)
    # % invariant. Where N(0,Sigma) is a zero-mean Gaussian distribution with
    # % covariance Sigma. Often L is a likelihood function in an inference problem.
    # %
    # % Inputs:
    # %              xx Dx1 initial vector (can be any array with D elements)
    # %
    # %      chol_Sigma DxD chol(Sigma). Sigma is the prior covariance of xx
    # %  or:
    # %    prior_sample Dx1 single sample from N(0, Sigma)
    # %
    # %     log_like_fn @fn log_like_fn(xx, varargin{:}) returns 1x1 log likelihood
    # %
    # % Optional inputs:
    # %    cur_log_like 1x1 log_like_fn(xx, varargin{:}) of initial vector.
    # %                     You can omit this argument or pass [].
    # %     angle_range 1x1 Default 0: explore whole ellipse with break point at
    # %                     first rejection. Set in (0,2*pi] to explore a bracket of
    # %                     the specified width centred uniformly at random.
    # %                     You can omit this argument or pass [].
    # %        varargin  -  any additional arguments are passed to log_like_fn
    # %
    # % Outputs:
    # %              xx Dx1 (size matches input) perturbed vector
    # %    cur_log_like 1x1 log_like_fn(xx, varargin{:}) of final vector

    # % Iain Murray, September 2009
    # % Tweak to interface and documentation, September 2010

    # % Reference:
    # % Elliptical slice sampling
    # % Iain Murray, Ryan Prescott Adams and David J.C. MacKay.
    # % The Proceedings of the 13th International Conference on Artificial
    # % Intelligence and Statistics (AISTATS), JMLR W&CP 9:541-548, 2010.

    umat = xx.u
    # nu, p, N = umat.shape
    v = xx.v;
    V = xx.V;
    S = xx.S;
    L = xx.L;
    x = xx.xx;
    flag = 0;
    fil = V.shape[0]

    D = x.size
    Ndatos = D / (v*fil);

    if cur_log_like is None:
        cur_log_like = log_lik_frob(S, V, 1)

    # Set up the ellipse and the slice threshold
    if prior.size == D and len(prior.shape) == 1:
        # % User provided a prior sample:
        nu = prior.reshape(x.shape)
    else:
        # % User specified Cholesky of prior covariance:
        if prior.shape != (D, D):
            raise ValueError('Prior must be given by a D-element sample or DxD chol(Sigma)');
        nu = np.reshape(prior.T.dot(np.random.normal(D)), x.shape);

    hh = 0.001 * np.log(np.random.uniform()) + cur_log_like;

    # % Set up a bracket of angles and pick a first proposal.
    # % "phi = (theta'-theta)" is a change in angle.
    if angle_range <= 0:
        # % Bracket whole ellipse with both edges at first proposed point
        phi = np.random.uniform()*2*np.pi;
        phi_min = phi - 2*np.pi;
        phi_max = phi;
    else:
        # % Randomly center bracket on current point
        phi_min = -angle_range * np.random.uniform();
        phi_max = phi_min + angle_range;
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min;

    cont = 0
    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference and check if it's on the slice
        # startOne = 1;
        # endOne = 0;
        xx_prop = np.real(x*np.cos(phi) + nu*np.sin(phi));
        umat = xx_prop.reshape((v, fil, Ndatos))
        # cell(v, fil);
        # for i in range(v):
        #     for j in range(fil):
        #         endOne = endOne + Ndatos;
        #         umat{i,j} = xx_prop(startOne:endOne);
        #         startOne = endOne + 1;

        V = GWP_construct(umat, L, v);
        cur_log_like = log_lik_frob(S,V, sigma2);

        if cur_log_like > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        if cont == 20:
            flag=1
            break;

        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            flag=1; break;
            # raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')

        # Propose new angle difference
        phi = np.random.uniform()*(phi_max - phi_min) + phi_min;
        cont += 1

    if flag == 0:
        xx['u'] = umat;
        xx['xx'] = xx_prop;
        xx['V'] = V;

    return xx, cur_log_like


def sample_hyper_kernel(ztau, sigma2prop, t, u, kern, muprior, sigma2prior):
    # % We do Metropolis-Hastings for sampling the posterior of the kernel
    # % hyperparameter. According to the paper, we use a lognormal distribution
    # % as the proposal
    # % Propose a sample

    zmean = np.log(ztau**2 / np.sqrt(sigma2prop+ztau**2));
    zsigma = np.sqrt(np.log(sigma2prop/ztau**2 + 1));
    zast = np.random.lognormal(zmean, zsigma);
    # Criterion to choose whether to accept the proposed sample or not
    logpzast = logpunorm(zast, t, u, kern, muprior, sigma2prior);
    qzastztau = lognpdf(zast, loc=zmean, scale=zsigma);

    logpztau = logpunorm(ztau, t, u, kern, muprior, sigma2prior);
    zmean = np.log(zast**2 / np.sqrt(sigma2prop+zast**2));
    zsigma = np.sqrt(np.log(sigma2prop / zast**2 + 1));
    qztauzast = lognpdf(ztau, loc=zmean, scale=zsigma);

    A = min(1, np.exp(logpzast - logpztau)*(qztauzast/qzastztau));
    # Now we decide whether to accept zast or use the previous value
    if np.random.uniform() < A:
        lp = zast;
        flag = 1;
    else:
        lp = ztau;
        flag = 0;
    return lp, flag


def logpunorm(l, t, umat, kern, muprior, sigma2prior):
    # kern.inverseWidth = l;
    kernel = kern(length_scale=1. / l)
    K = kernel(t[:,None])

    U = linalg.cholesky(K)
    invU = linalg.pinv(U)

    logDetK = fast_logdet(K);

    [v, p] = umat.shape
    umatinvU = np.array([umat[:, i].dot(invU) for i in range(p)])

    F = umatinvU.T
    f = F.ravel()
    logpugl = -0.5*((v*p)*logDetK) - 0.5 * f.dot(f) - 0.5 *f.size * np.log(2*np.pi);
    mean_prior = np.log(muprior**2 / np.sqrt(sigma2prior + muprior**2));
    sigma_prior = np.sqrt(np.log(sigma2prior / muprior**2 + 1));
    logpl = np.log(lognpdf(l, loc=mean_prior, scale=sigma_prior));
    logprob = logpugl + logpl;
    return logprob


def sample_L2(Ltau, sigma2Lprop, S, umat, sigma2error, muLprior, sigma2Lprior):
    # % We do Metropolis-Hastings for sampling the posterior of the elements in L.
    # % According to the paper, we use an spherical normal distribution
    # % as the proposal

    # % Run the MH individually per component of L
    free_elements = Ltau.size
    Lp = np.zeros(free_elements);
    for i in range(free_elements):
        Lp[i] = sample_L_comp(Ltau, i, sigma2Lprop[i], S, umat, sigma2error, muLprior[i], sigma2Lprior[i])
        Ltau[i] = Lp[i]

    return Lp

def sample_L_comp(Ltaug, i, sigma2Lprop, S, umat, sigma2error, muLprior, sigma2Lprior):
    # Propose a sample
    Ltau = Ltaug[i]
    Last = np.random.normal(Ltau, np.sqrt(sigma2Lprop));
    Lastg = Ltaug;
    Lastg[i] = Last;
    # Criterion to choose whether to accept the proposed sample or not
    logpLast = logpLpost(Lastg, i, S, umat, sigma2error, muLprior, sigma2Lprior);
    qzastztau = normpdf(Last, Ltau, np.sqrt(sigma2Lprop));

    logpLtau = logpLpost(Ltaug, i, S, umat, sigma2error, muLprior, sigma2Lprior);
    qztauzast = normpdf(Ltau, Last, np.sqrt(sigma2Lprop));

    A = min(1, np.exp(logpLast - logpLtau)*(qztauzast/qzastztau));

    # Now we decide whether to accept zast or use the previous value
    return Last if np.random.uniform() < A else Ltau

def logpLpost(Lv, i, S, umat, sigma2e, muprior, sigma2prior):
    L = np.zeros(4)
    L[np.tril_indices_from(L)] = Lv
    nu = umat.shape[0]
    D = GWP_construct(umat, L, nu);
    logpS = log_lik_frob(S, D, sigma2e);
    logpL = -(1/2)*np.log(2*np.pi*sigma2prior) - 0.5*(((Lv[i] - muprior)**2)/sigma2prior);
    return logpS + logpL;
