import numpy as np


def DescomposicionLU(A=np.ndarray):
    N = len(A)

    # Crear L0

    for i in range(N):
        # Crear la matriz diagonal
        m_diagonal = np.zeros(N)
        for j in range(N):
            if i == j:
                m_diagonal[i][j] = A[i,i]

        L = np.empty((N,N))
        L[i] = m_diagonal

        for k in range(N):


