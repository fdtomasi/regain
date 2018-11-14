function [w,q,lambda_tot] = glo_prox(w0,tau,blocks,weights,lambda0,tol,max_iter)
% GLO_PROX Computes the proximity operator of the group lasso with overlap
% penalty. Firts, identifies "active" blocks, then apply Bersekas's
% projected Newton method on the dual space.
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco


d = length(w0);
B = length(blocks);

% %------------NEW
% weights = zeros(B,1);
% for g = 1:B;
% %     weights(g) = length(blocks{g})*tau^2; %weight is sqrt{|g|}
%     weights(g) = tau^2; %not weighted
% end
% %------------NEW
weights = (weights.*tau).^2;



% if lambda is not initialized, then initialize it to 0
if isempty(lambda0);
    lambda0 = zeros(B,1);
end

beta = .5;
sigma = .1;
s_beta = 1;
epsilon = 0.001;

lambda_tot = zeros(B,1);

% Identify active blocks, by removing blocks such that w0 is already
% inside the corresponding cylinder
to_be_projected = zeros(B,1);
for g = 1:B;
    to_be_projected(g) = norm(w0(blocks{g}))^2>=weights(g);
end
to_be_projected = logical(to_be_projected);

%------------NEW
weights = weights(to_be_projected);
%------------NEW

blocks = blocks(to_be_projected);
B = length(blocks);
lambda = lambda0(to_be_projected);


q = 0;
if B==0;
    w = zeros(d,1);
    return
end

I  = zeros(d,B);
for g = 1:B;
    I(blocks{g},g) = 1;
end

% Bersekas constrained Newton method
stop =0;
i_null = 0;
while and(q<max_iter,~stop)
    q = q+1;
    lambda_prev = lambda;
    s = I*lambda_prev;
    denominator = 1./(1+s);
    grad = weights - I'*((w0.*denominator).^2);
    epsk = min(epsilon,norm(lambda-max(0,lambda-grad)));
    tmp = 2*w0.^2.*denominator.^3;

    I_inactive = find(or((grad<=0),(lambda>epsk)));
    n_inactive = length(I_inactive);
    B_inactive = zeros(n_inactive,n_inactive);

    for g = 1:n_inactive;
        B_inactive(g,g) = sum(tmp(blocks{I_inactive(g)}));
        for k = (g+1):n_inactive;
            B_inactive(g,k) = sum(tmp(intersect(blocks{I_inactive(g)},blocks{I_inactive(k)})));
            B_inactive(k,g) = B_inactive(g,k);
        end
    end
    p_inactive = pinv(B_inactive)*grad(I_inactive);

    I_active = find((grad>0).*(lambda<=epsk));
    n_active = length(I_active);
    if n_inactive==0;
        i_null=i_null+1;
    else
        i_null = 0;
    end
    if i_null>=5; break; end

    B_active = zeros(n_active,1);
    for g = 1:n_active;
        B_active(g) = sum(tmp(blocks{I_active(g)}));
    end
    p_active = grad(I_active,:)./B_active;

    x_inactive = grad(I_inactive)'*p_inactive;
    test = 1;
    m = 0;
    while test;
        m = m+1;
        step = beta^m*s_beta;
        lambda_m = zeros(B,1);
        lambda_m(I_active) = max(0,lambda_prev(I_active,:) - step*p_active);
        lambda_m(I_inactive) = max(0,lambda_prev(I_inactive,:) - step*p_inactive);
        s_m = I*lambda_m;
        fdiff = (w0.^2)'*(I*((lambda_m-lambda))./((1+s_m).*(1+s)))+sum(weights.*(lambda-lambda_m));
        x_active = grad(I_active)'*(lambda(I_active)-lambda_m(I_active));
        test = fdiff<sigma*(step*x_inactive+x_active);
    end
    lambda = lambda_m;
    stop = and(all(grad(lambda==0)>=0),all(abs(grad(lambda>0))<(tol)));

end

% given the solution of the dual problem, lambda, compute the primal solution w
s = I*lambda;
w = w0.*(1-1./(1+s));

lambda_tot(to_be_projected) = lambda;
