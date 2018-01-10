% build a toy data set with 3 groups of relevant variables. The variables
% in each group are noisy replicates of each other. Then apply 5-fold
% cross-validation with the algorithms:
% -l1 regularization (lasso)
% -l1l2 regularization (naive elastic net)
% -grouped l1 regularization (group lasso)

clear all
close all
addpath(genpath('../PASPAL'))


% fixing the seed of the random generators
seed=1;
randn('state',seed);
rand('twister',seed);

ng = 3; %number of relevant groups
dg = 5;%group size for relevant groups
n = 50; %number of training points
ntest = 1000;
d = 500; %total number of variables
fx = reshape(repmat(rand(ng,1)./ng,1,dg)',dg*ng,1); %true coefficient vector
% build ng relevant groups by generating dg noisy  replicates of each of the ng relevant variables
X = zeros(n+ntest,ng*dg);
for g = 1:ng;
    Xtmp = rand(n+ntest,1)*2-1;
    for i = ((g-1)*dg+1):(g*dg); 
        X(:,i) = awgn(Xtmp,10);
    end
end
Y = X*fx; %linear regression
Y = awgn(Y,10); %add noise to labels
% add irrelevant variables to reach d dimensions
X = [X rand(n+ntest,d-dg*ng)*2-1];

Xtest = X((n+1):(n+ntest),:);
Ytest = Y((n+1):(n+ntest));
X = X(1:n,:);
Y = Y(1:n);

protocol = 'both'; % if false performs  sparsity regularization followed by the rls de-biasing, if true performs also sparsity regularization alone
K = 5; % number of folds for K-fold cross validation

fprintf('\nDATA GENERATION MODEL\n')
fprintf('Number of variables:\t\t%d\n',d)
fprintf('Number of relevant variables:\t%d (%d groups of %d highly correlated variables)\n',dg*ng,ng,dg)
fprintf('Number of training samples:\t%d\n',n)
fprintf('Number of test samples:\t\t%d\n',ntest)
fprintf('\n')
fprintf('EXPERIMENTAL PROTOCOL\n')
fprintf('Validation protocol:\t%d-fold cross validation\n',K)
fprintf('Learning protocol:\ttwo protocols are employed\n')
fprintf('\t-`one_step` sparse regularizaton\n')
fprintf('\t-`two_step` sparse regularizaton + debiasing via Regularized Least Squares (RLS)\n')
%% L1 REGULARIZATION
fprintf('\n===================================================')
fprintf('\n\t\tL1 REGULARIZATION')
fprintf('\n===================================================\n')
[output_l1,model_l1] =l1l2_kcv(X,Y,'plot',true,'K',K,'smooth_par',0,'protocol',protocol);

% testing
prediction_l1 = l1l2_pred(model_l1,Xtest,Ytest,'regr');
disp(prediction_l1)

%% L1L2 REGULARIZATION
fprintf('\n===================================================')
fprintf('\n\t\tL1L2 REGULARIZATION')
fprintf('\n===================================================\n')
[output_l1l2,model_l1l2] =l1l2_kcv(X,Y,'plot',true,'K',K,'smooth_par',0.1,'protocol',protocol);

% testing
prediction_l1l2 = l1l2_pred(model_l1l2,Xtest,Ytest,'regr');
disp(prediction_l1l2)

%% GRL REGULARIZATION
fprintf('\n===================================================')
fprintf('\n\t\tGRL REGULARIZATION')
fprintf('\n===================================================\n')
% build blocks of 5 variables (1:,5, 6:10, etc.)
blocks = mat2cell(reshape(1:d,5,d/5),5,ones(d/5,1));
[output_grl,model_grl] =grl_kcv(X,Y,'blocks',blocks,'plot',true,'K',K,'smooth_par',0.01,'protocol',protocol);

% testing
prediction_grl = grl_pred(model_grl,Xtest,Ytest,'regr');
disp(prediction_grl)

