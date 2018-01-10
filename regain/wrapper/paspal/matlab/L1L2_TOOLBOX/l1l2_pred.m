function pred = l1l2_pred(model,Xtest,Ytest,err_type)
%L1L2_PRED Predicts outputs on test set
%
%   PRED = L1L2_PRED(MODEL,XTEST) Given MODEL from L1L2_KCV or a struct 
%   with fields BETA and OFFSET, predicts outputs on test set XTEST. 
%   PRED's fields:
%       -Y  (if MODEL contains field BETA)
%       -Y_1STEP(if MODEL contains field BETA_1STEP)
%       -Y_2STEPS(if MODEL contains field BETA_2STEPS)
%
%   PRED = L1L2_PRED(MODEL,XTEST,YTEST,ERR_TYPE) Given MODEL from 
%   L1L2_KCV predicts outputs and computes mean squared errror for test set XTEST. 
%   if ERR_TYPE='regr' compute mean squared errror for test set XTEST. If ERR_TYPE='class'
%   computes classification error. If ERR_TYPE = [w_pos, w_neg],
%   computes weighted classification error.
%   PRED's fields:
%       if MODEL contains field BETA
%           -Y:estimated outputs
%           -ERR: mean square error 
%       if MODEL contains field BETA_1STEP
%           -Y_1STEP: estimated outputs
%           -ERR_1STEP: mean square error
%       if MODEL contains field BETA_2STEPS
%           -Y_2STEPS:estimated outputs
%           -ERR_2STEPS: mean square error
%
%   See also L1L2_LEARN L1L2_KCV PREDICTION_ERROR
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

if nargin==3, err_type = 'regr'; end

% compute the predicted regression outputs (Y = X*beta + offset)
if isfield(model,'beta_1step')
    pred.y_1step = Xtest*model.beta_1step + model.offset_1step;
end
if isfield(model,'beta_2steps')
    pred.y_2steps = Xtest*model.beta_2steps + model.offset_2steps;
end
if isfield(model,'beta')
    pred.y = Xtest*model.beta + model.offset;
end

if nargin>2;    
    %estimates prediction errors
    if isfield(model,'beta_1step')    
        pred.err_1step = prediction_error(Ytest,pred.y_1step,err_type);
    end
    if isfield(model,'beta_2steps')    
        pred.err_2steps = prediction_error(Ytest,pred.y_2steps,err_type);
    end
    if isfield(model,'beta')    
        pred.err = prediction_error(Ytest,pred.y,err_type);
    end
end