import numpy as np

# Relajacion multivariable
def relajacion(F, x0, tol=1e-6, max_iter=20, verbose=True):
    x = np.array(x0, dtype=float)

    for k in range(max_iter):
        x_new = np.array(F(x), dtype=float)

        error = np.linalg.norm(x_new - x, ord=np.inf)

        if verbose:
            print(f"Iter {k+1:02d} -> x = {x_new}, error = {error:.6e}")

        if error < tol:
            print("\nConvergencia alcanzada")
            return x_new, k+1, True

        x = x_new

    print("\n No convergió")
    return x, max_iter, False



# parametros del problema

a = 1
b = 2



# Ssistema 1 reordenamiento malo
# x = y(a + x^2)
# y = b / (a + x^2)

def sistema_malo(x_vec):
    x, y = x_vec

    x_new = y * (a + x**2)
    y_new = b / (a + x**2)

    return [x_new, y_new]



# SISTEMA 2  BUENO
# x = (a*y)/(1 - x*y)
# y = b / (a + x^2)

def sistema_bueno(x_vec):
    x, y = x_vec

    # evitar división por cero
    if abs(1 - x*y) < 1e-10:
        return x_vec

    x_new = (a * y) / (1 - x * y)
    y_new = b / (a + x**2)

    return [x_new, y_new]



x0 = [1.0, 1.0]



print(" Sistema malo diverge")
relajacion(sistema_malo, x0)

print("\n\nSistema bueno(debe converger)")
sol, it, conv = relajacion(sistema_bueno, x0)



print("\nResultado final:")
print("Solución aproximada:", sol)
print("Iteraciones:", it)
print("Convergió:", conv)

print("\nSolución teórica:")
print("x =", b)
print("y =", b / (a + b**2))