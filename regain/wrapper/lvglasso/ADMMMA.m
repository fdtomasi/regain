function out = ADMMMA(emp_list,alpha,tau)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.continuation = 1; opts.num_continuation = 0;
opts.eta = sqrt(100); opts.muf = 1e-6;
opts.maxiter = 500; opts.stoptol = 1e-5;
opts.over_relax_par = 1;

R = cell(1, size(emp_list,3));
S = cell(1, size(emp_list,3));
L = cell(1, size(emp_list,3));
obj = cell(1, size(emp_list,3));
res = cell(1, size(emp_list,3));
iter = cell(1, size(emp_list,3));
tic
for i=1:size(emp_list,3)
    cov = emp_list(:,:,i);  %time is the last dimension
    n = size(cov,1);opts.mu = n;
    out_B = ADMM_B(cov,alpha,tau,opts);
    %%fprintf('ADMM_B: obj: %e, iter: %d, cpu: %3.1f \n',out_B.obj,out_B.iter,solve_B);
    R{i} = out_B.R;
    S{i} = out_B.S;
    L{i} = out_B.L;
    obj{i} = out_B.obj;
    res{i} = out_B.resid;
    iter{i} = out_B.iter;
end
out.elapsed_time = toc;
out.R = R;
out.S = S;
out.L = L;
out.iter = iter;
out.obj = obj;
out.res = res;
