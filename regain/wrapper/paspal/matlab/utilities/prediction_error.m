function err = prediction_error(Ytest,Ylearned,err_type)
%PREDICTION_ERROR Computes regression or classification error.
% 
%   [PRED] = PREDICTION_ERROR(YTEST,YLEARNED,ERR_TYPE) if ERR_TYPE='regr' 
%       compute mean squared errror for test set XTEST. If ERR_TYPE='class'
%       computes classification error. If ERR_TYPE = [w_pos, w_neg],
%       computes weighted classification error.
%
%   Copyright 2009-2010 Sofia Mosci and Lorenzo Rosasco

if isequal(err_type,'regr');
    %computes the mean squared error 
    err = norm(Ylearned-(Ytest))^2/length(Ytest);
else
    %computes the classification error 
    npos = sum(Ytest>0); %number of positive test samples
    nneg = sum(Ytest<0);%number of negative test samples
    if strcmp(err_type,'class'); 
        % the class weight will not be balanced, each error weight 1/n,
        % where n is the number of test samples
        err_type = [npos/(npos+nneg) nneg/(npos+nneg)]; 
    end
    class_fraction = err_type;    %uses input class weight for measuring positive and negative errors

    % estimates class labels by taking sign
    Ylearned = sign(Ylearned);
    FPrate = 0;
    FNrate = 0;
    if npos>0;
        % false negative (FN) rate
        FNrate = sum((Ylearned(Ytest>0)~=sign(Ytest(Ytest>0))))/npos;
    end
    if nneg>0;
        % false positive (FP) rate
        FPrate = sum((Ylearned(Ytest<0)~=sign(Ytest(Ytest<0))))/nneg;
    end
    %test error is weighted mean of FP and FN rate
    err = FNrate*max(class_fraction(1),nneg==0) + FPrate*max(class_fraction(2),npos==0);
end
