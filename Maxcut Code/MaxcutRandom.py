import cvxpy as cp
import numpy as np

def cut_size(A, T) :
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable and constraints
    n = A.shape[0]

    X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0]
    constraints += [ cp.diag(X) == 1]

    prob = cp.Problem(cp.Minimize(cp.trace(A @ X)), constraints)
    prob.solve(solver='CVXOPT')

    # Solve for the maximum cut
    U = np.linalg.cholesky(X.value)

    cut = 0
    for i in range(0, T) :
        r = np.random.normal(0, 1, n)
        y = np.sign(U @ r)

        # Calculate the cut
        cut = cut + (np.sum(A) - y.T@A@y)/4
    return round(cut / T)

# Generate a random SDP.
n = 20
p = 0.4
A = (np.random.rand(n, n) < p).astype(int)
A = np.triu(A) + np.triu(A).T
A = A - np.diag(np.diag(A))

cut = cut_size(A, 100)

print(cut)

