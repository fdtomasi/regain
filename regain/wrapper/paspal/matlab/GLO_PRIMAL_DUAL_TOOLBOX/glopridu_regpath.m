function [selected,tmin,n_iter,beta] = glopridu_regpath(X,Y,blocks,tau_values,varargin)
% GLOPRIDU_REGPATH group lasso regularization, path with the continuation method
% 
% [SELECTED] = GLOPRIDU_REGPATH(X,Y,TAU_VALUES) for each value in TAU_VALUES
%   evaluates l1 (smoothness parameter 0), and builds array of indices 
%   of selected features. The input data X is a NxD matrix, and the labels
%   Y are a Nx1 vector.
%   BLOCKS is either the number of blocks (with equal cardinality) or a cell array, 
%   where element i contains the indexes of the features in block i.
% 
% [SELECTED,TMIN] = GLOPRIDU_REGPATH(X,Y,TAU_VALUES) also returns the index of
%   the minimum acceptable  value (see below) for the sparsity parameter values in TAU_VALUES.
% 
% [SELECTED,TMIN,N_ITER] = GLOPRIDU_REGPATH(X,Y,TAU_VALUES) also returns a 
%   vector with the number of iterations for each value in TAU_VALUES
% 
% [SELECTED,TMIN,N_ITER,BETA] = GLOPRIDU_REGPATH(X,Y,TAU_VALUES) also returns 
%  the coefficients vector for each value in TAU_VALUES
% 
% GLOPRIDU_REGPATH(...,'PropertyName',PropertyValue,...) sets properties to the
%   specified property values.
%       -'weights': (default is ones(length(blocks),1)) weights for each
%        block
%       -'smooth_par': (default is 0) sets l2 parameter equal to SMOOTH_PAR
%       times the step size, which is internally computed
%       -'max_iter_ext': (default is 1e4) maximum number of outer
%       iterations
%       -'max_iter_int': (default is 1e4) maximum number of inner iterations
%       -'tolerance_ext': (default is 1e-6) tolerance for stopping the outer iterations.
%       -'tolerance_int': (default is 1e-6) tolerance for stopping the inner iterations.
% 
%   See also GLOPRIDU_ALGORITHM
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

   
if nargin<4; error('too few inputs!'); end
    
% DEFAULT PARAMETERS
smooth_par = 0;
max_iter_ext = 1e4;
max_iter_int = 1e2;
tol_ext = 1e-6;
tol_int = 1e-3;
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
        case 'max_iter_ext'
            max_iter_ext = args{i+1};
        case 'max_iter_int'
            max_iter_int = args{i+1};
        case 'tolerance_int'
            tol_int = args{i+1};
        case 'tolerance_ext'
            tol_ext = args{i+1};    
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

lambda0 = [];  %starting point for the first dual computation of the projection 

stop_par.tol_int = tol_int;
stop_par.tol_ext = tol_ext;
stop_par.max_iter_ext = max_iter_ext;
stop_par.max_iter_int = max_iter_int;

% compute solutions path  using  continuation strategy (warm restart)
for t = ntau:-1:1;
    % when smooth_par=0, exit if the algorithm selected more than n variables for the previously computed values of tau   
    % otherwise compute a new solution using previous solution  as
    % initialization
    if sparsity>=n;
        tmin = t+1;
        break
    else
        [beta_tmp,lambda,n_iter(t)] = glopridu_algorithm(X,Y,blocks,tau_values(t),weights,smooth_par,beta0,lambda0,sigma,stop_par);
        selected(:,t) = beta_tmp~=0; % selected variables
        sparsity = sum(selected(:,t)); % number of selected variables
        beta0 = beta_tmp; %update initialization vector
        lambda0 = lambda; %update initialization vector for dual computation of the projection
        % stores the coefficient vector only if it's requested as an output
        if nargout>3, beta{t} = beta_tmp; end
    end
end
