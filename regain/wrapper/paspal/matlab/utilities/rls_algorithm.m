function [beta] = rls_algorithm(X,Y,lambda)
%RLS_ALGORITHM Regularized Least Squares
%   BETA = RLS_ALGORITHM(X,Y) evaluates the Least Squares estimates of
%       ||Y-X*BETA||^2
%   BETA = RLS_ALGORITHM(X,Y,LAMBDA) evaluates the Regularized Least 
%   Squares estimates of 
%       1/N||Y-X*BETA||^2 +LAMBDA||BETA||^2
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

if nargin<3, lambda = 0; end
[n,d] = size(X);
if n<d;    beta = X'*pinv(X*X'+lambda*n*eye(n))*Y;
else    beta = pinv(X'*X+lambda*n*eye(d))*X'*Y;
end