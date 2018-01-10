function [beta,n_iter] = grl_algorithm(X,Y,blocks,tau,weights,smooth_par,beta0,sigma0,max_iter, tol)
% GRL_ALGORITHM Returns the minimizer of the empirical error penalized
% with group lasso penalty, solved via FISTA.
% 
%   [BETA] = GRL_ALGORITHM(X,Y,BLOCKS,TAU) returns the solution of the
%   group-lasso algorithm with sparsity parameter TAU (smoothness parameter 0).
%   BLOCKS is either the number of blocks (with equal cardinality), or a 
%   cell array, where element i contains the indexes of the features in 
%   block i.
%   If the input data X is a NxD matrix, and the labels Y are a Nx1 vector,
%   BETA is a Dx1 vector. 
%   The step size is A/N, where A is the  largest eigenvalue of X'*X and N is the number of training samples.
%   The algorithm stops when the value of the regularized empirical error reaches convergence.
% 
%   [BETA] = GRL_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS) returns the solution of the grl
%   regularization with sparsity parameter TAU, and weights WEIGTHS.
% 
%   [BETA] = GRL_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR) returns the solution of the grl
%   regularization with sparsity parameter TAU and smoothness parameter
%   A/N*SMOOTH_PAR. The step size is A/N*(1+SMOOTH_PAR).
% 
%   [BETA,K] = GRL_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR) returns also  the number of iterations.
%
%   [...] = GRL_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR,BETA0) uses BETA0 as
%   initialization of the iterative algorithm. If BETA0=[], sets BETA0=0
%
%   [...] = GRL_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR,BETA0,SIGMA0) sets the smoothness
%   parameter to SIGMA0*SMOOTH_PAR and and the step size to SIGMA0*(1+SMOOTH_PAR). 
%   If SIGMA0=[], sets the smoothness parameter to A/N*SMOOTH_PAR and the 
%   step size is A/N*(1+SMOOTH_PAR).
%
%   [...] = GRL_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR,BETA0,SIGMA0,MAX_ITER) the algorithm stops after
%   MAX_ITER iterations or when regularized empirical error reaches convergence 
%   (default tolerance is 1e-6). If MAX_ITER=[], sets MAX_ITER=1e5.
%
%   [...] = GRL_ALGORITHM(X,Y,BLOCKS,TAU,WEIGHTS,SMOOTH_PAR,BETA0,SIGMA0,MAX_ITER,TOL) uses TOL
%   as tolerance for stopping. 
%
%   See also GRL_REGPATH
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco



if nargin<4; error('too few inputs!'); end
if nargin<5; weights = []; end
if nargin<6; smooth_par = 0; end
if nargin<7; beta0 = []; end
if nargin<8; sigma0=[]; end
if nargin<9; max_iter = 1e5; end   
if nargin<10; tol = 1e-6; end
if nargin>10; error('too many inputs!'); end

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

%initialization for the values of the regularized empirical error (their
%mean will be compared with the value of the regularized empirical
%error at each iteration to establish if convergence is reached)
E_prevs = Inf*ones(10,1);

% group-l1l2 iterations

while and(n_iter<max_iter,~stop)

    n_iter = n_iter+1; %update the number of iterations
    beta_prev = beta; % update of the current iterate            
    Xb_prev = Xb;

    % computes the gradient step
    beta_noproj = h.*(1-mu_s) + XT*(Y-Xh);

    % apply soft-thresholding operator
    beta = cell(length(blocks),1);
    norm_beta = zeros(length(blocks));
    for j=1:length(blocks);
        norm_beta_j = norm(beta_noproj(blocks{j}));
        beta{j} = max(1-((tau_s.*weights(j))./norm_beta_j),0).*beta_noproj(blocks{j});
        norm_beta(j) = norm(beta_noproj(blocks{j}));
    end
    beta = cell2mat(beta);

    Xb = X*beta;

    t_new = .5*(1+sqrt(1+4*t^2)); %adaptive parameter used to combine the previous 2 iterates when building h
    h = beta + (t-1)/(t_new)*(beta-beta_prev); % combination of the 2 previous iterates
    Xh = Xb.*(1+ (t-1)/(t_new)) +(1-t)/(t_new).*Xb_prev;
    t = t_new;

    % evaluate the regularized empirical error on the current iterate
    E = norm(Xb-Y)^2/n + 2*tau*sum(norm_beta);  
    if smooth_par>0;
        E = E + mu*norm(beta)^2;
    end
    % compares the value of the regularized empirical error on the current iterate
    % with the mean of that of the previous 10 iterates times tol
    E_prevs(mod(n_iter,10)+1) = E;
    if (mean(E_prevs)-E)<mean(E_prevs)*tol; stop =1; end
end

