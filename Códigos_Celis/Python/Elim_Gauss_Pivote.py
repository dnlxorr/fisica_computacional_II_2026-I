import numpy as np

A = np.array([[0,1,4,1],
                    [3,4,-1,-1],
                    [1,-4,1,5],
                    [2,-2,1,3]],float)

v = np.array([-4,3,9,7],float)

N = len(v)
# Eliminaciòn Gaussiana

for fila in range(N):
    # Aplicar partial pivoting
    # Buscar el elemento más grande de la columna
    elem_max = fila
    for i in range(fila+1,N):
        if abs(A[i,fila])>abs(A[elem_max,fila]):
            elem_max = i

    if elem_max != fila:
        A[[fila,elem_max]] = A[[elem_max,fila]]
        #A[fila],A[elem_max] = A[elem_max],A[fila]
        v[fila], v[elem_max] = v[elem_max], v[fila]



    # Dividimos por el elemento de la diagonal
    div = A[fila,fila]
    A[fila,:] /= div
    v[fila] /= div

    # Ahora restamos de las filas inferiores

    for fila_inf in range(fila+1,N):
        mult = A[fila_inf,fila]
        A[fila_inf,:] -= mult * A[fila,:]
        v[fila_inf] -= mult * v[fila]

print(A)

# Backsubstitution

x = np.empty(N,float)

for fila in range(N-1,-1,-1):
    x[fila] = v[fila]
    for i in range(fila+1,N):
        x[fila] -= A[fila,i]*x[i]

print(list(range(N-1,-1,-1)))
print(x)