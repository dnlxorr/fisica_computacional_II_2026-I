import numpy as np
import time
from numpy.linalg import solve

# Eliminaciòn Gaussiana
def ElimGaussiana (A,v):
    N = len(v)
    for fila in range(N):
        # Dividimos por el elemento de la diagonal
        div = A[fila,fila]
        A[fila,:] /= div
        v[fila] /= div

        # Ahora restamos de las filas inferiores

        for fila_inf in range(fila+1,N):
            mult = A[fila_inf,fila]
            A[fila_inf,:] -= mult * A[fila,:]
            v[fila_inf] -= mult * v[fila]

    # Backsubstitution

    x = np.empty(N, float)

    for fila in range(N - 1, -1, -1):
        x[fila] = v[fila]
        for i in range(fila + 1, N):
            x[fila] -= A[fila, i] * x[i]

    return A,v,x


A = np.array([[2,1,0,0],
                    [3,4,-5,0],
                    [0,-4,3,5],
                    [0,0,1,3]],float)

v = np.array([1,1,1,1],float)


inicio = time.perf_counter()

A,v,x=ElimGaussiana(A,v)

fin = time.perf_counter()

incio2 = time.perf_counter()
u = solve(A,v)
fin2 = time.perf_counter()

print(A,end="\n")
print(v,end="\n")
print(x,end="\n")

print(u,end="\n")


print("Tiempo de ejecución: ",(fin-inicio)*1e6)
print("Tiempo de ejecuación con solve: ",(fin2-inicio)*1e6)
