import numpy as np
import matplotlib.pyplot as plt

# Número de puntos interiores
N = 20

# Dominio
a = 0
b = 1

# Paso
h = (b - a) / (N + 1)

# Vector de puntos (incluyendo frontera)
x = np.linspace(a, b, N + 2)

# Inicialización de la solución
u = np.zeros(N + 2)

# Condiciones de frontera
u[0] = 0
u[-1] = 0

# Lado derecho f(x)
f = - (np.pi ** 2) * np.sin(np.pi * x)

# Parámetro de relajación
w = 1.5

# Tolerancia y máximo de iteraciones
tol = 1e-6
max_iter = 10000

# Iteración SOR
for k in range(max_iter):

    error = 0

    # recorrer nodos interiores
    for i in range(1, N + 1):
        u_old = u[i]

        # fórmula SOR
        u[i] = (1 - w) * u[i] + (w / 2) * (u[i - 1] + u[i + 1] - h ** 2 * f[i])

        # cálculo del error
        error = max(error, abs(u[i] - u_old))

    if error < tol:
        print("Convergió en", k, "iteraciones")
        break

# Solución exacta
u_exact = np.sin(np.pi * x)

# Gráfica
plt.plot(x, u, label="Aprox SOR")
plt.plot(x, u_exact, '--', label="Exacta")
plt.legend()
plt.title("Solución de la EDO con SOR")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid()
plt.show()