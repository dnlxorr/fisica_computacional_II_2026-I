import numpy as np
import time

##Descompocicion LU para un sistema lineal de ecuaciones
# Matriz A
A = np.array([
    [2.0, -1.0, 1.0],
    [3.0, 3.0, 9.0],
    [3.0, 3.0, 5.0]
])

# Vector b
b = np.array([2.0, -1.0, 4.0])

N = len(A)
inicio= time.perf_counter()
# Matrices L y U
L = np.zeros((N,N))
U = np.zeros((N,N))


# hacemos la descompocicion LU


for i in range(N):

    # Calcular U
    for j in range(i, N):
        suma = 0
        for k in range(i):
            suma += L[i,k] * U[k,j]

        U[i,j] = A[i,j] - suma

    # Calcular L
    for j in range(i, N):

        if i == j:
            L[i,i] = 1
        else:
            suma = 0
            for k in range(i):
                suma += L[j,k] * U[k,i]

            L[j,i] = (A[j,i] - suma) / U[i,i]



# SUSTITUCION HACIA ADELANTE
# Ly = b


y = np.zeros(N)

for i in range(N):

    suma = 0
    for j in range(i):
        suma += L[i,j] * y[j]

    y[i] = b[i] - suma



# sustitucion inversa
# Ux = y


x = np.zeros(N)

for i in range(N-1, -1, -1):

    suma = 0
    for j in range(i+1, N):
        suma += U[i,j] * x[j]

    x[i] = (y[i] - suma) / U[i,i]


print("Matriz L:")
print(L)

print("\nMatriz U:")
print(U)

print("\nVector y:")
print(y)

print("\nSolucion x:")
print(x)

fin = time.perf_counter()
print("\ntime")
print((fin-inicio)*1e6)





