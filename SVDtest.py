import numpy as np
from numpy import linalg as LA

A = np.array([(6.0,1.0,1.0),(3.0,2.0,1.0),(1.0,0.0,2.0)])
B = np.array([(6.0,1.1,1.0),(3.2,2.0,1.0),(1.0,0.0,2.0)])
C = A-B
print C
er = LA.norm(C)
print(er)
A = np.delete(A, (1), axis = 0)
print(A)

r = np.genfromtxt("U.csv", delimiter = ",")
print(np.shape(r))
print(r)