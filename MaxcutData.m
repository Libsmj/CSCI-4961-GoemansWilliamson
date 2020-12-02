% Loads 10 datasets at a time and computs the cut for each of them
G = initialize;

for k = 1:10
    n = G{k}(1,1);
    e = G{k}(1,2);
    A = zeros(n,n);
    for t = 2:e+1
        i = G{k}(t,1);
        j = G{k}(t,2);
        A(i,j) = G{k}(t,3);
        A(j,i) = G{k}(t,3);
    end

    cut = gw_MaxCut(A, 1000);
    fprintf("%d,%d,%0.4f\n", G{k}(1,3), cut, cut/G{k}(1,3));
end


function G = initialize
    G = cell(10,1);
    for i = 1:10
        G{i} = load(append("MaxcutExamples\g05_100_", string(i-1), ".csv"));
    end
end

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