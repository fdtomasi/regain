function tau_max = glopridu_tau_max(X,Y,blocks,weights)
%GLOPRIDU_TAU_MAX estimates maximum value for sparsity parameter. 
%Values marger than tau_max yield null solutions.
% 
% [TAU_MAX] = GLOPRIDU_TAU_MAX(X,Y,BLOCKS) estimates maximum value for sparsity parameter 
%   for training set (X,Y). X is the NxD input matrix, and Y is the Nx1 
%   outputs vector
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

if nargin <4; weights = ones(length(blocks),1); end

tau_max = 1.1*norm(X'*Y)/(min(weights)*length(Y));
