clear all
close all
addpath(genpath('../PASPAL'))


% fixing the seed of the random generators
seed=1;
randn('state',seed);
rand('twister',seed);

drel = 25; %number of relevant variables
n = 100; %number of training points
d = 1000; %total number of variables
beta_true = [rand(drel,1); zeros(d-drel,1)]; %true coefficient vector
X = rand(n,d)*2-1;
Y = X*beta_true; %linear regression
Y = awgn(Y,20); %add noise to labels

% build test set
ntest = 200;
Xtest = rand(ntest,d)*2-1;
Ytest = Xtest*beta_true; %linear regression
Ytest = awgn(Ytest,20); %add noise to labels

protocol = 'both'; % if false performs  sparsity regularization followed by the rls de-biasing, if true performs also sparsity regularization alone
K = 5; % number of folds for K-fold cross validation

fprintf('\nDATA GENERATION MODEL\n')
fprintf('Number of variables:\t\t%d\n',d)
fprintf('Number of relevant variables:\t%d \n',drel)
fprintf('Number of training samples:\t%d\n',n)
fprintf('Number of test samples:\t\t%d\n',ntest)
fprintf('\n')
fprintf('EXPERIMENTAL PROTOCOL\n')
fprintf('Validation protocol:\t%d-fold cross validation\n',K)
fprintf('Learning protocol:\ttwo protocols are employed\n')
fprintf('\t-`one_step` sparse regularizaton\n')
fprintf('\t-`two_step` sparse regularizaton + debiasing via Regularized Least Squares (RLS)\n')


%%
% L1L2 REGULARIZATION
fprintf('\n========================================================')
fprintf('\n\t\tL1L2 REGULARIZATION')
fprintf('\n========================================================\n')

[output_l1l2,model_l1l2] =l1l2_kcv(X,Y,'plot',true,'K',K,'smooth_par',0.1,'protocol',protocol);

% evaluates selection errorselection_l1l2.FalseNeg_1step = 1-sum(model_l1l2.selected_1step(beta_true~=0))/(dg*ng);
selection_l1l2.FalseNeg_1step = sum(model_l1l2.selected_1step(beta_true==0))/(d-drel);
selection_l1l2.FalseNeg_2steps = 1-sum(model_l1l2.selected_2steps(beta_true~=0))/(drel);
selection_l1l2.FalseNeg_2steps = sum(model_l1l2.selected_2steps(beta_true==0))/(d-drel);

% testing
prediction_l1l2 = l1l2_pred(model_l1l2,Xtest,Ytest,'regr');
disp(selection_l1l2)
disp(prediction_l1l2)

%%
% GLOPRIDU
fprintf('\n========================================================')
fprintf('\n\t\tGLOPRIDU')
fprintf('\nwith blocks of 10 variables with 5 overlapping variables')
fprintf('\n========================================================\n')
% build blocks of 10 variables with 5 overlapping variables (1:10, 6:15, etc.)
blocks = mat2cell([reshape(1:d,10,d/10), reshape(6:(d-5),10,d/10 -1)],10,ones(d/10 +d/10 -1,1));

[output_glo,model_glo] =glopridu_kcv(X,Y,'blocks',blocks,'plot',true,'K',K,'protocol',protocol);

% evaluates selection error
selection_glo.FalseNeg_1step = 1-sum(model_glo.selected_1step(beta_true~=0))/(drel);
selection_glo.FalseNeg_1step = sum(model_glo.selected_1step(beta_true==0))/(d-drel);
selection_glo.FalseNeg_2steps = 1-sum(model_glo.selected_2steps(beta_true~=0))/(drel);
selection_glo.FalseNeg_2steps = sum(model_glo.selected_2steps(beta_true==0))/(d-drel);

% testing
prediction_glo = glopridu_pred(model_glo,Xtest,Ytest,'regr');

disp(selection_glo)
disp(prediction_glo)
