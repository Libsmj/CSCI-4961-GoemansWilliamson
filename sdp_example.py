import cvxpy as cp
import numpy as np

# Generate a random SDP.
n = 3
np.random.seed(2)
C = np.random.randn(n, n)

# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.
X = cp.Variable((n,n), symmetric=True)
# The operator >> denotes matrix inequality.
constraints = [X >> 0]
constraints += [
    cp.diag(X) == 1
]
prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
prob.solve(solver='CVXOPT')

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(X.value)