function out = ADMMMA(emp_cov, alpha, tau)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.continuation = 1; opts.num_continuation = 0;
opts.eta = sqrt(100); opts.muf = 1e-6;
opts.maxiter = 500; opts.stoptol = 1e-5;
opts.over_relax_par = 1;

n = size(emp_cov,1);
opts.mu = n;

tic; out = ADMM_B(emp_cov,alpha,tau,opts); out.elapsed_time = toc;
%%fprintf('ADMM_B: obj: %e, iter: %d, cpu: %3.1f \n',out_B.obj,out_B.iter,solve_B);
out.res = out.resid;
