import numpy as np
import math


def g(x):
     return np.array([(x[1]**2 + x[2] + 1)/3,
                      (x[0] + x[2]**2 + 1)/3,
                      (x[0] + x[1] + 1)/3
                      ])

# Relajacion multivariable

#def g(x):
 #   return np.array([np.sqrt((x[1] + 5)/2),
  #                   np.sqrt(x[0] + 1)
 #   ])

##implementamos la iteracion con relajacion

def relajacion(g, x0, lam=0.5, tol = 1e-8, max_iter=1000):
    x= np.array(x0, dtype=float)


    for i in range(max_iter):
        x_new = x + lam * (g(x) - x)

        print(f"Iter {i}: x = {x_new[0]:.6f}, y = {x_new[1]:.6f}, z = {x_new[2]:.6}")

        #criterio de convergencia
        if np.linalg.norm(x_new - x) < tol:
            print(f"convergio en {i} iteraciones")
            return x_new,i
        x = x_new


    print("no convergio")
    return x, max_iter


x0= np.array([0.3, 0.3, 0.3]) #punto inicial

solucion, it = relajacion(g, x0, lam=0.5)

print(f"Solución ≈ (x = {solucion[0]:.6f}, y = {solucion[1]:.6f}, z = {solucion[2]:.6}) en {it} iteraciones")








