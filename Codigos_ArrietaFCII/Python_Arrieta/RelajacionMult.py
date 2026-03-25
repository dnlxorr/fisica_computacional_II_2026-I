import numpy as np

from Codigos_ArrietaFCII.Python_Arrieta.metodoRelajacion1Variable import iteraciones

# Relajacion multivariable
def relajacion(F, x0, tol=1e-6, max_iter=1000):

    # Convertir a array de numpy
    x = np.array(x0, dtype=float)

    for k in range(max_iter):
        x_new = np.array(F(x), dtype=float)

        # Cálculo del error (norma infinita)
        error = np.linalg.norm(x_new - x, ord=np.inf)

        # Verificar convergencia
        if error < tol:
            return x_new, k + 1, True

        x = x_new

    # Si no converge
    return x, max_iter, False

def sistema_3var(x_vec):
    x, y, z = x_vec

    x_new = (y + z) / 3
    y_new = (x + z) / 3
    z_new = (x + y) / 3

    return [x_new, y_new, z_new]


print("\n=== Sistema 3 variables ===")

x0 = [1.0, 2.0, 3.0]

sol, it, conv = relajacion(sistema_3var, x0)

print("Solución:", sol)
print("Iteraciones:", it)
print("Convergió:", conv)


