import numpy as np


def Matrices_QR(A):

    N = len(A)

    R = np.zeros([N,N])

    q = np.zeros([N,N])
    u_normas = np.zeros([N])

    for i in range(N):
        u = A[:, i].copy()
        for j in range(i):
            proy = np.dot(q[:,j], A[:,i])
            u = u - proy * q[:,j]

        #Normalización
        norma = np.linalg.norm(u)
        u_normas[i] = norma
        q[:,i] = u / norma


    for i in range(N):
        R[i,i] = u_normas[i]
        for j in range(i+1,N):
            R[i,j] = np.dot(q[:,i],A[:,j])



    return q,R

def autovectores_autovalores(A):
    epsilon = 1e-10
    n = 5000
    N = len(A)
    V = np.zeros([N, N])
    for i in range(N):
        V[i, i] = 1.0

    for i in range(n):
        criterio = True
        Q, R = Matrices_QR(A)

        A = np.dot(R, Q)

        V = np.dot(V,Q)

        for i in range(N):
            if criterio == False:
                break
            for j in range(N):
                if i != j :
                    if A[i,j] > epsilon:
                        criterio = False
                        break


        # Criterio de convergencia
        #off_diag = A - np.diag(np.diag(A))
        #if np.linalg.norm(off_diag) < epsilon:
         #   break

        if criterio == True:
            break

    autovalores = np.diag(A)

    return autovalores,V,A






# *A = np.array([[1e-8,4e-8,8e-5,4e-9],
#               [4e-7,2e-3,3e-5,7e-8],
#               [8e-4,3e-5,6e-7,9e-2],
#               [4e-6,7e-6,9e-5,2e-5]])

A = np.array([[1,4,8,4],
              [4,2,3,7],
              [8,3,6,9],
              [4,7,9,2]])

R,Q = Matrices_QR(A)

print(R)
print(Q)

n = Q@R
print(n)

autovalores,V, a= autovectores_autovalores(A)

print("Autovalores\n")
print(autovalores)
print("Matriz A\n")
print(a)


