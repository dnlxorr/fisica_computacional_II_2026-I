import numpy as np

def Matrices_QR(A):
    # Descomposición QR usando Gram-Schmidt
    N = len(A)
    Q = np.zeros((N, N))
    R = np.zeros((N, N))

    for j in range(N):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def autovectores_autovalores(A):
    epsilon = 1e-10
    n = 5000
    N = len(A)

    V = np.zeros((N, N))
    for i in range(N):
        V[i, i] = 1.0

    for k in range(n):
        Q, R = Matrices_QR(A)

        A = np.dot(R, Q)
        V = np.dot(V, Q)

        # Criterio de convergencia
        off_diag = A - np.diag(np.diag(A))
        if np.linalg.norm(off_diag) < epsilon:
            break

    autovalores = np.diag(A)

    return autovalores, V, A


# Ejemplo de uso
A = np.array([[3.0, 4.0, 5.0],
              [2.0, 5.0, 10.0],
              [-4.0, 4.0, -3.0]])

autovalores, autovectores, A_final = autovectores_autovalores(A)

print("Autovalores:")
print(autovalores)

print("\nAutovectores (columnas):")
print(autovectores)

print("\nMatriz final (casi diagonal):")
print(A_final)

