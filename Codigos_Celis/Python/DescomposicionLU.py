import numpy as np

def DescomposicionLU(A):
    N = len(A)
    matrices = []
    matrices2 = []
    A_actual = A


    for c in range(N):
        # Crear L
        L = np.zeros([N, N])
        L_inversa = np.zeros([N, N])
        pivote_actual = A_actual[c, c]

        for i in range(N):
            L_inversa[i,i] = 1.0

        for i in range(N):
            # Crear la matriz diagonal
            L[i,i]=1.0

        for j in range(c+1,N):
            L[j,c] = -A_actual[j,c]/pivote_actual
            L_inversa[j,c] = A_actual[j,c]/pivote_actual

        matrices.append(L)
        matrices2.append(L_inversa)

        #Actualizacion de la matriz A
        A_actual = L @ A_actual # Será U

    L_total = np.zeros((N, N))
    for i in range(N):
        L_total[i,i] = 1.0

    for i in range(len(matrices2)):
        L_total = L_total @ matrices2[i]

    return matrices,A_actual,L_total

A = np.array([[4,-2,1],[20,-7,12],[-8,13,17]],float)
L,U,L_inv=DescomposicionLU(A)

for i,L in enumerate(L):
    print("L",i)
    print(L)

print("\n Matriz U:")
print(U)

print("\n Matriz L inv:")
print(L_inv)


A = L_inv @ U

print("\n",A)


