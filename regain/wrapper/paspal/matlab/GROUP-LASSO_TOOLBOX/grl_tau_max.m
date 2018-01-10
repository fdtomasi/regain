function tau_max = grl_tau_max(X,Y)
%GRL_TAU_MAX estimates maximum value for sparsity parameter. 
%Values marger than tau_max yield null solutions.
% 
% 	TAU_MAX = GRL_TAU_MAX(X,Y,BLOCKS) estimates maximum value for sparsity parameter 
%   for training set (X,Y). X is the NxD input matrix, and Y is the Nx1 
%   outputs vector
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

tau_max = 1.1*norm(X'*Y)/(length(Y));
