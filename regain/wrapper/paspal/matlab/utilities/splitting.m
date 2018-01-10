function sets = splitting(Y,K,rand_split)
%SPLITTING Splits data set in balanced subsets
%   SETS = SPLITTING(Y,K) return a cell array of K subsets of 1:n where
%       n=length(Y). The elements 1:n are split so that in each subset the
%       ratio between indices corresponding to positive and negative 
%       elements of Y is approximately as in the entire array Y. The 
%       subsets are obtained  by sequentially distributing the indices 1:n.
%
%   SETS = SPLITTING(Y,K,RAND_SPLIT) if RAND_SPLIT = false (default)
%       performs a deterministic split, otherwise, splits 1:n randomly,
%       though maintaining the class balance.
%       SPLITTING(Y,K,false) = SPLITTING(Y,K)
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

if nargin<2; error('too few inputs!'); end
if nargin==2, rand_split = false; end; 
if nargin>3; error('too many inputs!'); end

n = length(Y);

% put each element of 1:n in a different subset
if or(K==0,K==n);
    sets = cell(1,n);
    for i = 1:n, sets{i} = i; end
else
    
    %initialize the subsets
    sets = cell(1,K);
    
    if all(Y==round(Y)); % if it's a (multi)classification problem
        levels = unique(Y);
        for c = 1:length(levels);
                Ic = find(Y==levels(c));
            if rand_split;
                perm = randperm(length(Ic));
            else
                perm = 1:length(Ic);
            end
            i = 1;
            while i<=length(Ic);
                for v = 1:K;
                    if i<=length(Ic);
                        sets{v} = [sets{v}; Ic(perm(i))];
                        i = i+1;
                    end;
                end;
            end;
        end

    else
        c1 = find(Y>=0);
        c2 = find(Y<0);
        l1 = length(c1);
        l2 = length(c2);
        if rand_split;
            perm1 = randperm(l1);
            perm2 = randperm(l2);
        else
            perm1=1:l1;
            perm2=1:l2;
        end;
        i = 1;
        while i<=l1;
            for v = 1:K;
                if i<=l1;
                    sets{v} = [sets{v}; c1(perm1(i))];
                    i = i+1;
                end;
            end;
        end;
        i = 1;
        while i<=l2;
            for v = 1:K;
                if i<=l2;
                    sets{v} = [sets{v}; c2(perm2(i))];
                    i = i+1;
                end;
            end;
        end;
    end
end    