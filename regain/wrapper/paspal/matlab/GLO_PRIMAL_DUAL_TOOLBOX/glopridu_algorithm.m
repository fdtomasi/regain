function [beta,lambda,n_iter] = glopridu_algorithm(X,Y,blocks,tau,weights,smooth_par,beta0,lambda0,sigma0,stop_par)
% GLOPRIDU_ALGORITHM Returns the minimizer of the empirical error penalized
% with group lasso penalty, solved via FISTA.
%
%   [BETA] = GLOPRIDU_ALGORITHM(X,Y,BLOCKS,TAU) returns the solution of the
%   group-lasso algorithm with sparsity parameter TAU (smoothness parameter 0).
%   BLOCKS is either the number of blocks (with equal cardinality), or a
%   cell array, where element i contains the indexes of the features in
%   block i.
%   If the input data X is a NxD matrix, and the labels Y are a Nx1 vector,
%   BETA is a Dx1 vector.
%   The step size is A/N, where A is the  largest eigenvalue of X'*X and N is the number of training samples.
%   The algorithm stops when value of the regularized empirical error reaches convergence.
%
%   [BETA] = GLOPRIDU_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS) returns the solution of the glopridu
%   regularization with sparsity parameter TAU and weights WEIGHTS
%
%   [BETA] = GLOPRIDU_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,WEIGHTS,SMOOTH_PAR) returns the solution of the glopridu
%   regularization with sparsity parameter TAU and smoothness parameter
%   A/N*SMOOTH_PAR. The step size is A/N*(1+SMOOTH_PAR).
%
%   [BETA,K] = GLOPRIDU_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR) returns also  the number of iterations.
%
%   [...] = GLOPRIDU_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR,BETA0) uses BETA0 as
%   initialization of the iterative algorithm. If BETA0=[], sets BETA0=0
%
%   [...] = GLOPRIDU_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR,BETA0,LAMBDA0) uses LAMBDA0 as
%   initialization of the first computation of the projection. If LAMBDA0=[], sets LAMBDA0=0
%
%   [...] = GLOPRIDU_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR,BETA0,LAMBDA0,SIGMA0) sets the smoothness
%   parameter to SIGMA0*SMOOTH_PAR and and the step size to
%   SIGMA0*(1+SMOOTH_PAR).
%   If SIGMA0=[], sets the smoothness parameter to A/N*SMOOTH_PAR and the
%   step size is A/N*(1+SMOOTH_PAR).
%
%   [...] = GLOPRIDU_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR,BETA0,LAMBDA0,SIGMA0,STOP_PAR) stops
%   according to the fileds MAX_ITER_EXT, MAX_ITER_INT, TOL_EXT, TOL_INT in
%   STOP_PAR. If some of these fields are missing, then uses default
%   values.the algorithm stops after
%   The outer iteration stops when reaches MAX_ITER_EXT iterations or when the coefficient vector
%   error reaches convergencein l2 norm (default tolerance is TOL_EXT=1e-6).
%   If MAX_ITER_EXT=[], sets MAX_ITER_EXT=1e4.
%   The inner iteration stops when reaches MAX_ITER_INT iterations or when the coefficient vector
%   error reaches convergencein l2 norm (default tolerance is TOL_INT=1e-4).
%   If MAX_ITER_INT=[], sets MAX_ITER_INT=1e2.
%
%   See also GLOPRIDU_REGPATH.
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

if nargin<4; error('too few inputs!'); end
if nargin<5; weights = []; end
if nargin<6; smooth_par = 0; end
if nargin<7; beta0 = []; end
if nargin<8; lambda0 = []; end
if nargin<9; sigma0=[]; end
if nargin<10; stop_par = struct(); end
if nargin>10; error('too many inputs!'); end

blocks=transpose(blocks);

if isfield(stop_par,'max_iter_ext');
    max_iter_ext = stop_par.max_iter_ext;
else
    max_iter_ext = 1e4;
end
if isfield(stop_par,'max_iter_int');
    max_iter_int = stop_par.max_iter_int;
else
    max_iter_int = 1e2;
end
if isfield(stop_par,'tol_ext');
    tol_ext = stop_par.tol_ext;
else
    tol_ext = 1e-6;
end
if isfield(stop_par,'tol_int');
    tol_int = stop_par.tol_int;
else
    tol_int = 1e-4;
end

[n,d] =size(X);

% if only the number of blocks is given, build blocks of equal
% cardinality with sequential features
if ~iscell(blocks);
    blocks = num2cell(reshape(1:d,d/blocks,blocks),1);
end

% if sigma is not specified in input, set  it to as a/n
if isempty(sigma0);
    sigma0 = normest(X)^2/n; %step size for smooth_par=0
end

% if weights are not specified in input, set them to 1
if isempty(weights);
    weights = ones(length(blocks),1);
end

mu = smooth_par*sigma0; % smoothness parameter is rescaled

sigma = sigma0+mu; % step size

% useful normalization that avoid computing the same computations for
% each iteration
mu_s = mu/sigma;
tau_s = tau/sigma;
XT = X'./(n*sigma);


% initialization
if isempty(beta0);
    beta0 = zeros(d,1);
end
n_iter = 0; %the number of iterations
stop=0; %logical variable for stopping the iterations
beta = beta0; % initialization for iterate n_iter-1
h = beta0; % initialization for combination of the previous 2 iterates (iteratations n_iter_1 and n_iter-2)
t = 1; %initialization for the adaptive parameter used to combine the previous 2 iterates when building h
% precomputes X*beta and X*h to avoid computing them twice
Xb = X*beta;
Xh = Xb;
lambda_prev = lambda0; %initialization for the dual vector in computing the projection


% GLOPRIDU iterations

while or(and(n_iter<max_iter_ext,~stop),n_iter<2)

    n_iter = n_iter+1; %update the number of iterations
    beta_prev = beta; % update of the current iterate
    Xb_prev = Xb;

    % computes the gradient step
    %beta_noproj = h.*(1-mu_s) + XT*(Y-Xh);

    % compute the proximity operator with tolerance depending on k
    [beta,q,lambda] = glo_prox(h.*(1-mu_s) + XT*(Y-Xh),tau_s,blocks,weights,lambda_prev,tol_int*n_iter^(-3/2),max_iter_int);
    lambda_prev = lambda; %update initialization for dual vector lambda (warm starting)
    Xb = X*beta;

    t_new = .5*(1+sqrt(1+4*t^2)); %adaptive parameter used to combine the previous 2 iterates when building h
    h = beta + (t-1)/(t_new)*(beta-beta_prev); % combination of the 2 previous iterates
    Xh = Xb.*(1+ (t-1)/(t_new)) +(1-t)/(t_new).*Xb_prev;
    t = t_new;

    stop = norm(Xb-Xb_prev)<=norm(Xb_prev)*tol_ext;
end
