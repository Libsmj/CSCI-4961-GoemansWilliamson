%% Set up a random graph
clear;clc;
n = 50;
p = 0.4;
A = triu(rand(n)<p);
A = A+A';
%% Use CVX
cvx_begin
    variable X(n,n) symmetric
    minimize trace(A*X)
        diag(X) == ones(n,1);
        X == semidefinite(n);
cvx_end
%%
U=chol(X);
r = randn(n,1);
x_hat = sign(U'*r);
cut = (sum(A(:)) - x_hat'*A*x_hat)/4;
SDP_opt = (sum(A(:)) - trace(A*X))/4;