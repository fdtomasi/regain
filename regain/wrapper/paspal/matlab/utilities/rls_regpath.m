function [beta] = rls_regpath(X,Y,lambda_values)
% RLS_REGPATH rls solutions for a set of values of the parameter
% 
% [BETA] = RLS_REGPATH(X,Y,LAMBDA_VALUES) for each value in LAMBDA_VALUES
%   evaluates rls solution, and builds cell of solution
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

[n,d] = size(X);
beta = cell(length(lambda_values),1);
if n<d;    
    [U,S,V] = svd(X*X');
    for l = 1:length(lambda_values);
        beta{l} = X'*U*diag((diag(S)+lambda_values(l)*n).^-1)*V'*Y;
    end
else
    [U,S,V] = svd(X'*X);
    for l = 1:length(lambda_values);
        beta{l} = U*diag((diag(S)+lambda_values(l)*n).^-1)*V'*X'*Y;
    end
end
