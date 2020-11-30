%% Set up a random graph
n = 20;
rng(2)

p = 0.4;
A = rand(n) < p;
A = triu(A) + triu(A,1)';
A = A - diag(diag(A));
disp(A)

% Use CVX
cvx_begin quiet
    variable X(n,n) symmetric
    minimize trace(A*X)
        diag(X) == ones(n,1);
        X == semidefinite(n);
cvx_end

%%
U = chol(X);
r = mvnrnd(zeros(n,1),diag(ones(n,1)))';
y = sign(U*r);
cut = (sum(A(:)) - y'*A*y)/4;
disp(cut)

% SDP_opt = (sum(A(:)) - trace(A*X))/4;