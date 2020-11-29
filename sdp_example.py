from numpy import matrix
from Irene import sdp
b = [1, -1, 1]
C = [matrix([[-33, 9], [9, -26]]),
     matrix([[-14, -9, -40], [-9, -91, -10], [-40, -10, -15]])]
A1 = [matrix([[7, 11], [11, -3]]),
      matrix([[21, 11, 0], [11, -10, -8], [0, -8, -5]])]
A2 = [matrix([[-7, 18], [18, -8]]),
      matrix([[0, -10, -16], [-10, 10, 10], [-16, 10, -3]])]
A3 = [matrix([[2, 8], [8, -1]]),
      matrix([[5, -2, 17], [-2, 6, -8], [17, -8, -6]])]
SDP = sdp('cvxopt')
SDP.SetObjective(b)
SDP.AddConstantBlock(C)
SDP.AddConstraintBlock(A1)
SDP.AddConstraintBlock(A2)
SDP.AddConstraintBlock(A3)
SDP.solve()
print(SDP.Info)