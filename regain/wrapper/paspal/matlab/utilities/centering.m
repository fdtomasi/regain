function [Xcentered,Ycentered,meanX,meanY] = centering(X,Y,center)
%CENTERING Centering of a data set
%   [XCENTERED,YCENTERED] = CENTERING(X,Y,CENTER) if CENTER=true centers a 
%       data set X,Y where X is a matrix NxD and Y an array Nx1 and returns 
%       the cenetered matrix XCENTERED, and array YCENTERED. Matrix X is 
%       centered column by column by subtracting the column mean. 
%       Y is centered by  subtracting its mean. 
%       If CENTER=false, XCENTERED=X, YCENTERED=Y,
%   [XCENTERED,YCENTERED,MEANX,MEANY] = CENTERING(X,Y,CENTER) also returns
%       the columns means of matrix X, MEANX, and the mean of vector Y,
%       MEANY. if CENTER=false MEANX is a vector of zeros and meanY = 0.
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

[n,xdim] = size(X);
[n,ntasks] = size(Y);

if center;
    % computes means
    meanX = mean(X); 
    meanY = mean(Y);
else
    % save fake null means
    meanX = zeros(1,xdim);
    meanY= zeros(1,ntasks);
end

Xcentered = X-repmat(meanX,n,1);
Ycentered = Y-repmat(meanY,n,1);
