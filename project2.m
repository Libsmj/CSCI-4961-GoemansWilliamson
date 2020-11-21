%% max_cut_sdp.m
% clear;clc;
% n=10; %number of nodes
% A=zeros(n,n); %adjacency matrix
% edges=[[1,2];[2,3];[3,4];[4,5];[5,1];...
%        [1,6];[2,7];[3,8];[4,9];[5,10];...
%        [6,8];[6,9];[7,9];[7,10];[8,10]];
% A(sub2ind(size(A),edges(:,1),edges(:,2)))=1;
% A=A+A';

%% Set up a random graph
n = 50;

p = 0.4;
A = rand(n) < p;
A = triu(A) + triu(A,1)';
A = A - diag(diag(A));


cvx_begin sdp quiet
    variable X(n,n) symmetric
    minimize(-trace( A* (ones(n,n)-X ))/4)
    subject to
    diag(X)==ones(n,1)
    X>=0
    X == semidefinite(n);
cvx_end

V = chol(X); %Factor
w = rand(n,1); %Random vector
y = sign(V*w); %Partition data

%Output size of cut
fprintf("Size of cut: %d\n",trace(A * (1 - y*y'))/4);