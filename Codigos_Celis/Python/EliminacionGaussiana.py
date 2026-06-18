import numpy as np

def eliminacionGaussiana(A,v):
    N = len(v)
    # Eliminaciòn Gaussiana

    for fila in range(N):
        # Dividimos por el elemento de la diagonal
        div = A[fila, fila]
        A[fila, :] /= div
        v[fila] /= div

        # Ahora restamos de las filas inferiores

        for fila_inf in range(fila + 1, N):
            mult = A[fila_inf, fila]
            A[fila_inf, :] -= mult * A[fila, :]
            v[fila_inf] -= mult * v[fila]


    x = np.empty(N, float)

    for fila in range(N - 1, -1, -1):
        x[fila] = v[fila]
        for i in range(fila + 1, N):
            x[fila] -= A[fila, i] * x[i]

    return x

A = np.array([[3.0,4.0,5.0,2.0,3.0],
        [2.0,5.0,10.0,11.7,2.0],
        [-4.0,4.0,-3.0,3.9,3.0],
        [2.0,6.0,-7.0,9.0,10.0],
        [3.0,5.0,7.0,8.0,5.0]],float)

v = np.array([3.0,8.0,5.0,3.5,2.0],float)

x = eliminacionGaussiana(A,v)

print(x)
