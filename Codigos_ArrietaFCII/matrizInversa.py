import numpy as np

#calculo matriz inversa procedimiento completo

A = np.array([
    [2, 1, 1],
    [1, 3, 2],
    [1, 0, 0]
])

def matriz_inversa(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # llamamos la Matriz identidad
    I = np.eye(n)

    A_inv = np.zeros((n, n))

    # Resolver A x = e_i para cada columna
    for i in range(n):

        #se selecciona la columna i de la identidad
        e = I[:, i]
    #resuelve el sistema
        x = np.linalg.solve(A, e)


# guardar solución
        A_inv[:, i] = x

    return A_inv

A_inv = matriz_inversa(A)

print("Matriz A:")
print(A)

print("\nInversa de A:")
print(A_inv)

print("\nVerificación A * A_inv = I:")
print(A @ A_inv)