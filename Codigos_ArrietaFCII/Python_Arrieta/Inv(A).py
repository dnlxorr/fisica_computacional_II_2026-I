from numpy.linalg import inv
import numpy as np
#calculo matriz inversa solo usando la libreria numpy.linalg import inv
#x=inv(A)
A = np.array([[2,1,1,5,6],
              [1,3,2,6,3],
              [1,0,0,7,8],
              [1,4,6,7,8],
              [12,3,5,6,8]])
print("matriz A original")
print(A)
X = inv(A)
print("matriz inversa calculada A^{-1}: ")
print(X)