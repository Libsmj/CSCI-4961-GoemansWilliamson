%% Set up a random graph
rng(2)
n = 50;

p = 0.4;
A = rand(n) < p;
A = triu(A) + triu(A,1)';
A = A - diag(diag(A));

%% Use CVX
cvx_begin quiet
    variable X(n,n) symmetric
    minimize trace(A*X)
        diag(X) == ones(n,1);
        X == semidefinite(n);
cvx_end
%%
U = chol(X);
r = mvnrnd(zeros(n,1),diag(ones(n,1)))';
x_hat = sign(U*r);
cut = (sum(A(:)) - x_hat'*A*x_hat)/4;
SDP_opt = (sum(A(:)) - trace(A*X))/4;