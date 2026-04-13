"""
Ejercicio 6.7 - Cadena de resistores

Red de resistores iguales (R) entre un riel V+ = 5V y tierra (0V).
Se buscan los voltajes V1...VN en los nodos internos.

"""

import numpy as np
from numpy import copy
import matplotlib.pyplot as plt

def eliminacionGaussiana(A, v):
    N = len(v)

    for fila in range(N):
        div = A[fila, fila]
        A[fila, :] /= div
        v[fila] /= div

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


def construccion_sistema(N, Vmas=5.0):
    """
    Construye la matriz A (NxN) densa y el vector w para N nodos internos.
    Solo se usa para N pequeño (parte b).
    """
    A = np.zeros((N, N))
    w = np.zeros(N)

    for i in range(N):
        if i == 0:
            A[0, 0] = 3
            if N > 1: A[0, 1] = -1
            if N > 2: A[0, 2] = -1
            w[0] = Vmas
        elif i == 1:
            A[1, 0] = -1
            A[1, 1] = 4
            if N > 2: A[1, 2] = -1
            if N > 3: A[1, 3] = -1
            w[1] = Vmas
        elif i == N - 2:
            A[i, i-2] = -1
            A[i, i-1] = -1
            A[i, i]   =  4
            A[i, i+1] = -1
        elif i == N - 1:
            if N > 2: A[i, i-2] = -1
            A[i, i-1] = -1
            A[i, i]   =  3
        else:
            A[i, i-2] = -1
            A[i, i-1] = -1
            A[i, i]   =  4
            A[i, i+1] = -1
            A[i, i+2] = -1

    return A, w


def construccion_sistema_banded(N, Vmas=5.0):
    """
    Construye la matriz en formato de banda (5 x N) y el vector w.
    Nunca crea la matriz densa NxN.

    Formato: ab[u + i - j, j] = A[i,j], con u=2.
    Fila 0 → diagonal +2
    Fila 1 → diagonal +1
    Fila 2 → diagonal  0  (principal)
    Fila 3 → diagonal -1
    Fila 4 → diagonal -2
    """
    u = 2
    ab = np.zeros((5, N))
    w  = np.zeros(N)

    for j in range(N):
        ab[u, j] = 3.0 if (j == 0 or j == N - 1) else 4.0

        for off in [1, 2]:
            if j - off >= 0:
                ab[u - off, j] = -1.0   # superdiagonales
            if j + off < N:
                ab[u + off, j] = -1.0   # subdiagonales

    w[0] = Vmas
    w[1] = Vmas

    return ab, w


def banded(Aa, va, up=2, down=2):
    A = copy(Aa)
    v = copy(va)
    N = len(v)

    # ---- Eliminación hacia adelante ----
    for m in range(N):
        div = A[up, m]
        v[m] /= div

        for k in range(1, down + 1):
            if m + k < N:
                v[m + k] -= A[up + k, m] * v[m]

        for i in range(up):
            j = m + up - i
            if j < N:
                A[i, j] /= div
                for k in range(1, down + 1):
                    A[i + k, j] -= A[up + k, m] * A[i, j]


    for m in range(N - 2, -1, -1):
        for i in range(up):
            j = m + up - i
            if j < N:
                v[m] -= A[i, j] * v[j]

    return v


# ------------------------------------------------------------------
# Parte b) N = 6
# ------------------------------------------------------------------
N = 6
A6, w6 = construccion_sistema(N)
V6 = eliminacionGaussiana(A6, w6)

print("=" * 40)
print("Parte b) N = 6")
print("=" * 40)
for i, v in enumerate(V6):
    print(f"  V{i+1} = {v:.6f} V")
print(f"  Todos en [0, 5] V: {np.all((V6 >= 0) & (V6 <= 5))}")

# ------------------------------------------------------------------
# Parte c) N = 10 000  (matriz en banda)
# ------------------------------------------------------------------
N_big = 10_000
A10k, w10k = construccion_sistema_banded(N_big)   # solo 5 x N, no NxN
V_big = banded(A10k, w10k)

print(f"\n{'=' * 40}")
print(f"Parte c) N = {N_big}")
print(f"{'=' * 40}")
print(f"  V1        = {V_big[0]:.6f} V")
print(f"  V2        = {V_big[1]:.6f} V")
print(f"  V{N_big//2}     = {V_big[N_big//2]:.6f} V")
print(f"  V{N_big-1}  = {V_big[-2]:.6f} V")
print(f"  V{N_big}    = {V_big[-1]:.6f} V")
print(f"  Min: {V_big.min():.4f} V  Max: {V_big.max():.4f} V")
print(f"  Todos en [0, 5] V: {np.all((V_big >= 0) & (V_big <= 5))}")

# ------------------------------------------------------------------
# Gráficas
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(1, N+1), V6, color='steelblue', edgecolor='white')
axes[0].axhline(5, color='red',   ls='--', lw=1, label='V+ = 5 V')
axes[0].axhline(0, color='black', ls='--', lw=1, label='Tierra = 0 V')
axes[0].set_xlabel("Nodo i")
axes[0].set_ylabel("Voltaje (V)")
axes[0].set_title("Ejercicio 6.7 b)  N = 6")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, N_big+1), V_big, color='steelblue', lw=0.5)
axes[1].axhline(5, color='red',   ls='--', lw=1)
axes[1].axhline(0, color='black', ls='--', lw=1)
axes[1].set_xlabel("Nodo i")
axes[1].set_ylabel("Voltaje (V)")
axes[1].set_title(f"Ejercicio 6.7 c)  N = {N_big}")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ejercicio_6_7.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfica guardada como ejercicio_6_7.png")