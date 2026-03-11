import numpy as np
import time
# Matriz
A = np.array([
    [2.0, 1.0, -1.0, -3.0, 2.0],
    [-3.0, -1.0, 2.0, 5.0, 1.0],
    [-2.0, 1.0, 2.0, 8.0, 4.0],
    [-2.0, 3.0, 4.0, 3.0, 5.0],
    [-5.0, 7.0, 1.0, 4.0, 9.0]
])
# Vector de terminos independientes
b = np.array([8.0, -11.0, -3.0, 3.0, 6.0])

A_original = A.copy()
b_original = b.copy()

N= len(b)



inicio= time.perf_counter()




print("matriz oiginal")
print(A.copy())
print("\n vector de terminos independientes")
print(b.copy())

# ELIMINACIÓN GAUSSIANA

#Para especificar i:fila pivote, j: fila a eliminar, k: columna actualizada.

for i in range(N):

    for j in range(i+1, N):

        multiplicador = A[j][i] / A[i][i]

        for k in range(i, N):
            A[j][k] = A[j][k] - multiplicador * A[i][k]

        b[j] = b[j] - multiplicador*b[i]

print("\nMatriz triangular superior obtenida:")
print(A)

print("\n matriz de terminos independientes obtenida")
print(b)



# SUSTITUCIÓN INVERSA
x = np.zeros(N)
for i in range(N-1, -1, -1):

    suma = 0

    for j in range(i+1, N):
        suma += A[i][j]*x[j]

    x[i] = (b[i] - suma) / A[i][i]


fin = time.perf_counter()
print("\ntime")
print((fin-inicio)*1e6)
print("Vector solución del sistema obtenido:")
print(x)