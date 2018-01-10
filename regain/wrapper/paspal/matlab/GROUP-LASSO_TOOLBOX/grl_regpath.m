function [selected,tmin,n_iter,beta] = grl_regpath(X,Y,blocks,tau_values,varargin)
% GRL_REGPATH group lasso regularization, path with the continuation method
% 
% [SELECTED] = GRL_REGPATH(X,Y,TAU_VALUES) for each value in TAU_VALUES
%   evaluates l1 (smoothness parameter 0), and builds array of indices 
%   of selected features. The input data X is a NxD matrix, and the labels
%   Y are a Nx1 vector.
%   BLOCKS is either the number of blocks (with equal cardinality) or a cell array, 
%   where element i contains the indexes of the features in block i.
% 
% [SELECTED,TMIN] = GRL_REGPATH(X,Y,TAU_VALUES) also returns the index of
%   the minimum acceptable  value (see below) for the sparsity parameter values in TAU_VALUES.
% 
% [SELECTED,TMIN,N_ITER] = GRL_REGPATH(X,Y,TAU_VALUES) also returns a 
%   vector with the number of iterations for each value in TAU_VALUES
% 
% [SELECTED,TMIN,N_ITER,BETA] = GRL_REGPATH(X,Y,TAU_VALUES) also returns 
%  the coefficients vector for each value in TAU_VALUES
% 
% GRL_REGPATH(...,'PropertyName',PropertyValue,...) sets properties to the
%   specified property values.
%       -'weights': (default is ones(length(blocks),1)) weights for each
%        block
%       -'smooth_par': (default is 0) sets l2 parameter equal to SMOOTH_PAR
%       times the step size, which is internally computed
%       -'max_iter': (default is 1e5) maximum number of iterations
%       -'tolerance': (default is 1e-6) tolerance for stopping the iterations.
% 
%   See also GRL_ALGORITHM
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

   
    if nargin<4; error('too few inputs!'); end
    
    % DEFAULT PARAMETERS
smooth_par = 0;
max_iter = 1e5;
tol = 1e-6;
weights = ones(length(blocks),1);

% OPTIONAL PARAMETERS
args = varargin;
nargs = length(args);
for i=1:2:nargs
    switch args{i},
		case 'weights'
            weights = args{i+1};
		case 'smooth_par'
            smooth_par = args{i+1};
		case 'max_iter'
            max_iter = args{i+1};
		case 'tolerance'
            tol = args{i+1};
    end
end

ntau = length(tau_values); %number of values for TAU
[n,d] = size(X);

%compute step size
sigma = normest(X)^2/n;

% initialize the coefficients vector only if it's requested as an output
if nargout>3;
    beta = cell(ntau,1);
end
selected = ones(d,ntau); %initialization for the logical vector of selected variables
n_iter = zeros(ntau,1);  %initialization for the numbers of iterations
sparsity = 0; %initialization for the number of selected variables

% initialization for the index of the minimum acceptable value for the 
% sparsity parameter in TAU_VALUES
tmin = 1; 

beta0 = zeros(d,1); %starting point in the iterative algorithm for largest value of the sparsity parameter

% compute solutions path  using  continuation strategy (warm restart)
for t = ntau:-1:1;
    % when smooth_par=0, exit if the algorithm selected more than n variables for the previously computed values of tau   
    % otherwise compute a new solution using previous solution  as initialization
    if and(smooth_par==0,sparsity>=n);
        tmin = t+1;
        break
    else
        [beta_tmp,n_iter(t)] = grl_algorithm(X,Y,blocks,tau_values(t),weights,smooth_par,beta0,sigma,max_iter,tol);
        selected(:,t) = beta_tmp~=0; % selected variables
        sparsity = sum(selected(:,t)); % number of selected variables
        beta0 = beta_tmp; %update initialization vector
        % stores the coefficient vector only if it's requested as an output
        if nargout>3, beta{t} = beta_tmp; end
    end
end
