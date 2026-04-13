import numpy as np


## parametros

R1 = R3 = R5 = 1e3       # 1 kΩ
R2 = R4 = R6 = 2e3       # 2 kΩ
C1 = 1e-6                 # 1 µF
C2 = 0.5e-6               # 0.5 µF
xp = 3.0                  # V+ = 3 V
w  = 1000.0               # ω = 1000 s⁻¹

#admitancia nodales terminos diagonales
Z1 = 1/R1 + 1/R4 + 1j*w*C1

Z2= 1/R2 + 1/R5 + 1j*w*C1 + 1j*w*C2

Z3 = 1/R3 + 1/R6 + 1j*w*C2

# terminos de acoplamiento cpacitivo
# la corriente que fluye entre nodos vecinos a través de los capacitores

iC1 = 1j*w*C1
iC2 = 1j*w*C2

#Matriz A (compleja tiragonal 3x3)

A= np.array([
    [Z1,   iC1 ,  0],
    [-iC1,  Z2, -iC2],
    [0,   -iC2,   Z3]
], dtype= complex)

# vector terminos independientes.
#corriente inyectada en cada nodo por la fuente V+
b= np.array([xp/R1, xp/R2, xp/R3], dtype=complex)


# implementamos la descomposicion LU
def lu_descomposicion(M):

    n= len(M)
    U= M.astype(complex).copy()
    L= np.eye(n, dtype= complex)
    P = list (range(n))

    for k in range (n):
        ## hacemos pivoteo para encontrat la fila con mayor elemento
        filaMx = k + np.argmax(np.abs(U[k:, k]))

        if filaMx != k:
            U[[k, filaMx]]= U[[filaMx,k]] #intercambiar filas en U
            L[[k, filaMx], :k] = L[[filaMx, k], :k] #intercambio en L
            P[k], P[filaMx] = P[filaMx],P[k] # guarda el intercambio

        #eliminacion hacia abajo

        for i in range (k+1, n):

            multiplicador = U[i,k]/U[k,k]
            L[i,k]= multiplicador ##se guarda en la matriz L
            U[i,k:] -= multiplicador * U[k,k:] # se anula(0) en U

    return L, U, P

def sustitucionHaciaAdelante(L, b_perm):
    n = len(b_perm)
    y = np.zeros(n, dtype=complex)
    for i in range(n):
        y[i] = b_perm[i] - np.dot(L[i, :i], y[:i])
    return y

def sustitucionHaciaAtras(U, y):
    n = len(y)
    x = np.zeros(n, dtype=complex)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x


L, U, P = lu_descomposicion(A)
b_perm = b[P]
y = sustitucionHaciaAdelante(L, b_perm)
x = sustitucionHaciaAtras(U, y)

print("=" * 45)
print("   SOLUCIÓN — Ejercicio 6.5 (LU)")
print("=" * 45)
print(f"{'Voltaje':<10} {'Amplitud (V)':>14} {'Fase (°)':>12}")
print("-" * 45)
for k, xk in enumerate(x, start=1):
    amp = np.abs(xk) #modulo en numero complejo
    phase = np.degrees(np.angle(xk)) #argumento en grados
    print(f"  V{k:<8} {amp:>14.6f} {phase:>11.4f}°")

print("=" * 45)

# Verificación: comparar con numpy.linalg.solve
x_ref = np.linalg.solve(A, b)
print("\nVerificación con numpy.linalg.solve:")
print(f"  Diferencia máxima: {np.max(np.abs(x - x_ref)):.2e}")
