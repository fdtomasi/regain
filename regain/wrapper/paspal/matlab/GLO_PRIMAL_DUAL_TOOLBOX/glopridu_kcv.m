function [cv_output,model] = glopridu_kcv(X,Y,varargin)
%GLOPRIDU_KCV Parameters choice through cross validation for the group lasso algorithm
%(variable selector) followed by Regularized Least Squares (for debiasing).
%   CV_OUTPUT = GLOPRIDU_KCV(X,Y) Given training set, (X,Y), performs leave-one-out cross
%   validation for GLOPRIDU algorithm. GLOPRIDU is used for selection of the
%   variables, whereas the regression coefficients are evaluated on the selected variables
%   via RLS. The input data X is a NxD matrix, and the output vector  Y is Nx1
%   [CV_OUTPUT,MODEL] = GLOPRIDU_KCV(X,Y) Also returns the estimated model
%
%   GLOPRIDU_KCV(...,'PropertyName',PropertyValue,...) sets properties to the
%   specified property values.
%       -'blocks': number of blocks (with equal cardinality) or a cell array,
%        where element i contains the indexes of the features in block i.
%       -'weights': (default is ones(length(blocks),1)) weights for each
%        block, it can be either a string ('constant' for unitary weights,
%        or 'sqrt_dim' for weight equal to the square root of the number of
%        elements in each group), or an numeric array.
%       -'L1_n_par': number of values for the sparsity parameter (default is 100)
%       -'L1_max_par': maximum value for the sparsity parameter (default is
%        chosen automatically, see paper)
%       -'L1_min_par': minimum value for the sparsity parameter (default is
%        chosen automatically as L1_max_par/100)
%       -'L1_pars': vector of values for the sparsity parameter. When not specified,
%        100 values are chosen completely automatically, or using
%        'L1_n_par','L1_max_par', and 'L1_min_par'.
%       -'RLS_pars': values for RLS parameter. When not specified,
%        50 values are chosen automatically.
%       -'smooth_par': value of the smoothing parameter (default is 0)
%       -'err_type': 'regr'(deafult) for regression, 'class' for
%        classfication
%       -'protocol': (default is 'two_steps') if 'one_step' computes error
%        of the model learned via GLOPRIDU without RLS-debiasing; if
%        'two_steps' computes error of the model learned via GLOPRIDU with
%        RLS-debiasing; if 'both' computes both.
%       -'offset': (default is true) add unpenalized offset to the model.
%       -'K': specify number K of folds in in  K-fold cross-validation.
%        If K=0 or K=length(Y) it performs LOO cross-validation
%       -'rand_split': if false (default) perfoms a deterministic split of
%       the data, if true the split is random.
%       -'plot': (default is false) if true plots training,  validation
%        errors, and number of selected variables vs the sparsity parameter.
%
%   CV_OUTPUT's fields
%	-sparsity(array of double): number of selected features for each value of the sparsity parameter
% 	-selected_all(cell array): indices of the selected features for each value of the sparsity parameter.
%   if 'protocol'=='one_step':
%       -tau_opt_1step(double): sparsity parameter minimizing the K-fold
%        cross-validation error for the 1-step framework (GLOPRIDU only)
%       -err_KCV_1step(array of double): cross-validation error on validation set for for
%        the 1-step framework
%       -err_train_1step(array of double): cross-validation error on training set for the
%        1-step framework
%   if 'protocol'=='two_steps':
%       -tau_opt_2steps(double): sparsity parameter minimizing the K-fold
%        cross-validation error for the 2-steps framework (GLOPRIDU and RLS)
%       -lambda_opt_2steps(double): RLS parameter minimizing the K-fold
%        cross-validation error for the 2-steps framework
%       -err_KCV_2steps(2d array of double): cross-validation error on validation set for the
%        2-steps framework
%       -err_train_2steps(2d array of double): cross-validation error on training set for the
%        2-steps framework
%  if 'protocol'=='both': has both of the above sets of fields.
%
%   MODEL's fields
%   -offset: offset to be added to the estimated model
%   if 'protocol'=='one_step':
%       -selected_1step: indexes of the selected features for the optimal
%        parameters  for the 1-step framework
%       -beta_1step: coefficient vector for the optimal parameters for the
%        1-step framework
%   if 'protocol'=='two_steps':
%       -selected_2steps: indexes of the selected features for the optimal
%        parameters for the 2-steps framework
%       -beta_2steps: coefficient vector for the optimal parameters  for
%        the 2-steps framework
%  if 'protocol'=='both': has both of the above sets of fields.
%
%   See also GLOPRIDU_REGPATH, RLS_REGPATH, GLOPRIDU_LEARN
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco


if nargin<2, error('too few input!'), end

% DEFAULT PARAMETRS
err_type = 'regr';
smooth_par = 0;
ntau = 50;
tau_min = [];
tau_max = [];
tau_values = [];
lambda_values = [];
K = 0;
split = false;
offset = true;
plotting = false;
blocks = num2cell(ones(size(X,2),1)); %if blocks are not given, perform kcv for l1 (l1l2) regularization
weights = 'constant';
withRLS = true;
woRLS = false;

% OPTIONAL PARAMETERS
args = varargin;
nargs = length(args);
for i=1:2:nargs
    switch args{i},
        case 'blocks'
            blocks = args{i+1};
        case 'weights'
            weights = args{i+1};
        case 'L1_pars'
            tau_values = args{i+1};
        case 'L1_min_par'
            tau_min = args{i+1};
        case 'L1_max_par'
            tau_max = args{i+1};
        case 'L1_n_par'
            ntau = args{i+1};
        case 'RLS_pars'
            lambda_values = args{i+1};
        case 'err_type'
            err_type = args{i+1};
        case 'smooth_par'
            smooth_par = args{i+1};
        case 'K'
            K = args{i+1};
        case 'rand_split'
            split = args{i+1};
        case 'offset'
            offset = args{i+1};
        case 'plot'
            plotting = args{i+1};
        case 'protocol'
            if strcmp(args{i+1},'one_step');
                withRLS = false;
                woRLS = true;
            elseif strcmp(args{i+1},'two_steps');
                withRLS = true;
                woRLS = false;
            elseif strcmp(args{i+1},'both');
                withRLS = true;
                woRLS = true;
            else
                error('Unknown protocol!!!')
            end
    end
end

if ischar(weights);
    switch weights
        case 'constant'
            weights = ones(length(blocks),1);
        case 'sqrt_dim'
            weights = zeros(length(blocks),1);
            for g = 1:length(blocks);
                weights(g) = sqrt(length(blocks{g}));
            end
    end
end

% center data by subtracting means (if offset=true)
if or(or(isempty(tau_values),isempty(tau_max)),isempty(lambda_values))
    [Xtmp,Ytmp] = centering(X,Y,offset);
end

% Sparsity parameter
% if values for the sparsity parameter are not given in input (with
% 'L1_pars'), builds geometric series
if isempty(tau_values);
    if isempty(tau_max);
        % estimate the maximum value of the sparsity parameter (larger
        % values should produce null solutions)
        tau_max = glopridu_tau_max(Xtmp,Ytmp,weights);
    end
    % set tau_min
    if isempty(tau_min);
        tau_min = tau_max/100;
    end
    % geometric series
    tau_values = [tau_min tau_min*((tau_max/tau_min)^(1/(ntau-1))).^(1:(ntau-1))]; %geometric series.
else
    ntau = length(tau_values);
end

% RLS parameters for debiasing
if and(isempty(lambda_values),withRLS);
    sigma = normest(Xtmp*Xtmp');
    lambda_values = sigma*(10.^(-9.8:.2:0));
end


if or(or(isempty(tau_values),isempty(tau_max)),isempty(lambda_values))
    clear Xtmp Ytmp
end


sets = splitting(Y,K,split); %splits the training set in K subsets

% initialization
% without debias step
if woRLS
    err_KCV = ones(length(sets),ntau).*var(Y);
    err_train = ones(length(sets),ntau).*var(Y);
end
% with debias step
if withRLS
    err_KCV2 = ones(length(sets),ntau,length(lambda_values)).*var(Y);
    err_train2 = ones(length(sets),ntau,length(lambda_values)).*var(Y);
end
selected = cell(length(sets),1);
sparsity = zeros(length(sets),ntau);
n_iter = zeros(ntau,length(sets));


for i = 1:length(sets);
    ind = setdiff(1:length(Y),sets{i}); %indices of training set

    % centering (needed to compute offset)
    [Xtr,Ytr,meanX,meanY] = centering(X(ind,:),Y(ind),offset);
    Xts = X(sets{i},:);
    Yts = Y(sets{i});
    % evaluate all betas for all taus at the same time
    [selected{i},tmin,n_iter(:,i),beta] = glopridu_regpath(Xtr,Ytr,blocks,tau_values,'weights',weights,'smooth_par',smooth_par);
    selected{i} = selected{i}~=0;
    sparsity(i,:) = sum(selected{i});
    % for each value of the sparsity parameter, use the group lasso solution for
    % selection and train rls on the selected features, then evaluate error
    % on validation set (err_KCV)
    for t = tmin:ntau;
        if woRLS
            model.offset = meanY-meanX*beta{t};
            model.beta = beta{t};
            % evaluates prediction error on validation set
            pred = glopridu_pred(model,Xts,Yts,err_type);
            err_KCV(i,t) = pred.err;
            % when evaluating training error, offset must not be added
            % because training data is already centered
            model.offset = 0;
            % evaluates prediction error on training set
            pred = glopridu_pred(model,Xtr,Ytr,err_type);
            err_train(i,t) = pred.err;
            clear model pred;
        end

        if withRLS
            beta_rls = rls_regpath(Xtr(:,selected{i}(:,t)),Ytr,lambda_values);
            for j = 1:length(lambda_values);
                model.offset = meanY-meanX(selected{i}(:,t))*beta_rls{j};
                model.beta = beta_rls{j};
                % evaluates prediction error on validation set
                pred = glopridu_pred(model,Xts(:,selected{i}(:,t)),Yts,err_type);
                err_KCV2(i,t,j) = pred.err;
                % when evaluating training error, offset must not be added
                % because training data is already centered
                model.offset = 0;
                % evaluates prediction error on training set
                pred = glopridu_pred(model,Xtr(:,selected{i}(:,t)),Ytr,err_type);
                err_train2(i,t,j) = pred.err;
                clear model pred;
            end
        end

    end

end

n_iter = mean(n_iter,2);
% save outputs
cv_output.selected_all = selected;
cv_output.sparsity = mean(sparsity);
clear selected sparsity

if woRLS
    % evaluate avg. error over the splits
    err_KCV = reshape(mean(err_KCV,1),ntau,1);
    err_train = reshape(mean(err_train,1),ntau,1);

    % find sparsity parameter minimizing the error
    t_opt = find(err_KCV==min(err_KCV),1,'last');
    cv_output.tau_opt_1step = tau_values(t_opt);
    cv_output.err_KCV_1step = err_KCV;
    cv_output.err_train_1step = err_train;
end

if withRLS
    % evaluate avg. error over the splits
    err_KCV2 = reshape(mean(err_KCV2,1),ntau,length(lambda_values));
    err_train2 = reshape(mean(err_train2,1),ntau,length(lambda_values));

    % for each value of the sparsity parameter, find rls parameter minimizing the
    % error
    lambda_opt = zeros(ntau,1);
    err_KCV_opt2 = zeros(ntau,1);
    err_train_opt2 = zeros(ntau,1);
    for t = 1:ntau;
        l_opt = find(err_KCV2(t,:)==min(err_KCV2(t,:)),1,'last');
        lambda_opt(t) = lambda_values(l_opt);
        err_KCV_opt2(t) = err_KCV2(t,l_opt);
        err_train_opt2(t) = err_train2(t,l_opt);
    end

    % find sparsity parameter minimizing the error
    t_opt2 = find(err_KCV_opt2==min(err_KCV_opt2),1,'last');

    cv_output.tau_opt_2steps = tau_values(t_opt2);
    cv_output.lambda_opt_2steps = lambda_opt(t_opt2);
    cv_output.err_KCV_2steps = err_KCV2;
    cv_output.err_train_2steps = err_train2;
end


% Compute final model using parameters found by KCV. 
if nargout>1;
    if woRLS;
        [model.beta_1step,model.offset_1step] = glopridu_learn(X,Y,blocks,tau_values(t_opt:end),'weights',weights,'smooth_par',smooth_par);
        model.selected_1step = model.beta_1step~=0;
    end
    if withRLS;
        [model.beta_2steps,model.offset_2steps] = glopridu_learn(X,Y,blocks,tau_values(t_opt2:end),'weights',weights,'smooth_par',smooth_par,'RLS_par',lambda_opt(t_opt2));
        model.selected_2steps = model.beta_2steps~=0;
    end
end


% plot KCV error, training error, number of selected variables and number of iterations for varying tau. 
if plotting;
    if smooth_par==0;
        figure('Name','L1 regularization')
    else
        figure('Name','GLOPRIDU regularization')
    end
    c = 0;
    if woRLS
        c = c+1;
        subplot(withRLS+woRLS+2,1,c)
        semilogx(tau_values(tmin:end),err_train(tmin:end),'bs-','MarkerSize',3,'MarkerFaceColor','b'); hold on;
        semilogx(tau_values(tmin:end),err_KCV(tmin:end),'rs-','MarkerSize',3,'MarkerFaceColor','r');
        legend('train','validation');
        xlim = get(gca,'Xlim');
        ylim = get(gca,'Ylim');
        semilogx(xlim,repmat(min(err_train),2,1),'b:')
        semilogx(xlim,repmat(min(err_KCV),2,1),'r:')
        semilogx(repmat(tau_values(t_opt),2),[ylim(1) min(err_KCV)],'r:')
        title('CV error without RLS');
    end
    if withRLS;
        c = c+1;
        subplot(withRLS+woRLS+2,1,c)
        semilogx(tau_values(tmin:end),err_train_opt2(tmin:end),'bs-','MarkerSize',3,'MarkerFaceColor','b'); hold on;
        semilogx(tau_values(tmin:end),err_KCV_opt2(tmin:end),'rs-','MarkerSize',3,'MarkerFaceColor','r');
        legend('train','validation');
        xlim = get(gca,'Xlim');
        ylim = get(gca,'Ylim');
        semilogx(xlim,repmat(min(err_train_opt2), 2,1),'b:')
        semilogx(xlim,repmat(min(err_KCV_opt2), 2,1),'r:')
        semilogx(repmat(tau_values(t_opt2),2),[ylim(1) min(err_KCV_opt2)],'r:')
        title('CV error with RLS');
    end
    c = c+1;
    subplot(withRLS+woRLS+2,1,c)
    semilogx(tau_values(tmin:end),cv_output.sparsity(tmin:end),'ks-','MarkerSize',3,'MarkerFaceColor','b');
    hold on
    if woRLS
        semilogx([xlim(1) tau_values(t_opt)],repmat(cv_output.sparsity(t_opt),2,1),'r:')
        semilogx(repmat(tau_values(t_opt),2),[ylim(1) cv_output.sparsity(t_opt)],'r:')
    end
    if withRLS
        semilogx([xlim(1) tau_values(t_opt2)],repmat(cv_output.sparsity(t_opt2),2,1),'r:')
        semilogx(repmat(tau_values(t_opt2),2),[ylim(1) cv_output.sparsity(t_opt2)],'r:')
    end
    xlabel('\tau');
    title('# of selected variables');

    c = c+1;
    subplot(withRLS+woRLS+2,1,c)
    semilogx(tau_values(tmin:end),n_iter(tmin:end),'ks-','MarkerSize',3,'MarkerFaceColor','b');
    hold on
    if woRLS
        semilogx([xlim(1) tau_values(t_opt)],repmat(n_iter(t_opt),2,1),'r:')
        semilogx(repmat(tau_values(t_opt),2),[ylim(1) n_iter(t_opt)],'r:')
    end
    if withRLS
        semilogx([xlim(1) tau_values(t_opt2)],repmat(n_iter(t_opt2),2,1),'r:')
        semilogx(repmat(tau_values(t_opt2),2),[ylim(1) n_iter(t_opt2)],'r:')
    end
    xlabel('\tau');
    title('# of iterations');
end