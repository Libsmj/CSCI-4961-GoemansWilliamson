% Set up a random graph
n = 20;

p = 0.4;
A = rand(n) < p;
A = triu(A) + triu(A,1)';
A = A - diag(diag(A));
%disp(A)

cut = gw_MaxCut(A, 1000);
fprintf("Size of cut %d\n", cut)


function cut = gw_MaxCut(A, T)
    [n,~] = size(A);
    % Use CVX
    cvx_begin quiet
        variable X(n,n) symmetric
        minimize trace(A*X)
            diag(X) == ones(n,1);
            X == semidefinite(n);
    cvx_end

    %%
    U = chol(X);
    
    cut = 0;
    for i = 1:T
        r = mvnrnd(zeros(n,1),diag(ones(n,1)))';
        y = sign(U*r);
        cut = cut + (sum(A(:)) - y'*A*y)/4;
    end
    cut = round(cut / T);
    
end