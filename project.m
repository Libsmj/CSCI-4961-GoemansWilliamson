%% Set up a random graph
n = 100;

% a = zeros(125-1,1);
% for n = 2:125

    p = 0.4;
    A = rand(n) < p;
    A = triu(A) + triu(A,1)';
    A = A - diag(diag(A));

    % Use CVX
    tic
    cvx_begin quiet
        variable X(n,n) symmetric
        minimize trace(A*X)
            diag(X) == ones(n,1);
            X == semidefinite(n);
    cvx_end
%     a(n-1) = toc;
% end
% hold on
% xlim([2 125])
% plot(2:125, a)
% ylabel("cpu time");
% xlabel("n");
% hold off

%%
U = chol(X);
r = mvnrnd(zeros(n,1),diag(ones(n,1)))';
y = sign(U*r);
cut = (sum(A(:)) - y'*A*y)/4
SDP_opt = (sum(A(:)) - trace(A*X))/4;