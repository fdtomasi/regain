function [beta,n_iter] = l1l2_algorithm(X,Y,tau,smooth_par,beta0,sigma0,max_iter,tol)
% L1L2_ALGORITHM Returns the minimizer of the empirical error penalized
% with l1 or l1l2 penalties, solved via FISTA.
%
%   BETA = L1L2_ALGORITHM(X,Y,TAU) returns the solution of the l1
%   regularization algorithm with sparsity parameter TAU (smoothness
%   parameter 0) If the input data X is a NxD matrix, and the labels Y are 
%   a Nx1 vector, BETA is a Dx1 vector. The step size is A/N, where A is 
%   the largest eigenvalue of X'*X and N is the number of training samples.
%   The algorithm stops when the value of the regularized empirical error 
%   reaches convergence.
% 
%   BETA = L1L2_ALGORITHM(X,Y,TAU,SMOOTH_PAR) returns the solution of the l1l2
%   regularization with sparsity parameter TAU and smoothness parameter
%   A/N*SMOOTH_PAR. The step size is A/N*(1+SMOOTH_PAR).
% 
%   [BETA,N_ITER] = L1L2_ALGORITHM(X,Y,TAU,SMOOTH_PAR) returns also  the number of iterations.
%
%   [...] = L1L2_ALGORITHM(X,Y,TAU,SMOOTH_PAR,BETA0) uses BETA0 as
%   initialization of the iterative algorithm. If BETA0=[], sets BETA0=0
%
%   [...] = L1L2_ALGORITHM(X,Y,TAU,SMOOTH_PAR,BETA0,SIGMA0) sets the smoothness
%   parameter to SIGMA0*SMOOTH_PAR and and the step size to SIGMA0*(1+SMOOTH_PAR). 
%   If SIGMA0=[], sets the smoothness parameter to A/N*SMOOTH_PAR and the 
%   step size is A/N*(1+SMOOTH_PAR).
%
%   [...] = L1L2_ALGORITHM(X,Y,TAU,SMOOTH_PAR,BETA0,SIGMA0,MAX_ITER) the algorithm stops after
%   MAX_ITER iterations or when regularized empirical error reaches convergence 
%   (default tolerance is 1e-6). If MAX_ITER=[], sets MAX_ITER=1e5.
%
%   [...] = L1L2_ALGORITHM(X,Y,TAU,SMOOTH_PAR,BETA0,SIGMA0,MAX_ITER,TOL) uses TOL
%   as tolerance for stopping. 
%
%   See also L1L2_REGPATH.
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

if nargin<3; error('too few inputs!'); end
if nargin<4; smooth_par = 0; end
if nargin<5; beta0 = []; end
if nargin<6; sigma0=[]; end
if nargin<7; max_iter = 1e5; end
if nargin<8; tol = 1e-6; end
if nargin>8; error('too many inputs!'); end

[n,d] =size(X);

% if sigma is not specified in input, set  it to as a/n
if isempty(sigma0);
    sigma0 = normest(X)^2/n; %step size for smooth_par=0
end

mu = smooth_par*sigma0; % smoothness parameter is rescaled

step = sigma0+mu; % step size

% useful normalization that avoid computing the same computations for
% each iteration
mu_s = mu/step;
tau_s = tau/step;
XT = X'./(n*step);

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

% l1l2 iterations

while and(n_iter<max_iter,~stop)

    n_iter = n_iter+1; %update the number of iterations
    beta_prev = beta; % update of the current iterate            
    Xb_prev = Xb;

    % computes the gradient step
    beta_noproj = h.*(1-mu_s) + XT*(Y-Xh);

    % apply soft-thresholding operator
    beta = beta_noproj.*max(0,1-tau_s./abs(beta_noproj));

    Xb = X*beta;

    t_new = .5*(1+sqrt(1+4*t^2)); %adaptive parameter used to combine the previous 2 iterates when building h
    h = beta + (t-1)/(t_new)*(beta-beta_prev); % combination of the 2 previous iterates
    Xh = Xb.*(1+ (t-1)/(t_new)) +(1-t)/(t_new).*Xb_prev;
    t = t_new;

    % evaluate the regularized empirical error on the current iterate
    E = norm(Xb-Y)^2/n + 2*tau*sum(abs(beta));  
    if smooth_par>0;
        E = E + mu*norm(beta)^2;
    end
    % compares the value of the regularized empirical error on the current iterate
    % with the mean of that of the previous 10 iterates times tol
    E_prevs(mod(n_iter,10)+1) = E;
    if (mean(E_prevs)-E)<mean(E_prevs)*tol; stop =1; end
end