import numpy as np

# Sistema Ax = b
A = np.array([[4.0, -1.0, -6.0,0.0],
              [-5.0, -4.0, 10.0,8.0],
              [0.0, 9.0, 4.0,-2.0],
              [1.0,0.0,-7.0,5.0]])

b = np.array([2.0,21.0,-12.0,-6.0])

# Parámetros
omega = 0.5
n = len(b)
x = np.zeros(n)

tolerancia = 1e-8
error = 1.0

while error > tolerancia:
    x_antigua = x.copy()

    for i in range(n):

        suma1 = 0.0
        for j in range(i):
            suma1 += A[i,j] * x[j]

        suma2 = 0.0
        for j in range(i+1, n):
            suma2 += A[i,j] * x_antigua[j]

        x[i] = (1 - omega)*x_antigua[i] + (omega/A[i,i])*(b[i] - suma1 - suma2)
        print(x[i])
    error = np.linalg.norm(x - x_antigua)

print("Solución:", x)