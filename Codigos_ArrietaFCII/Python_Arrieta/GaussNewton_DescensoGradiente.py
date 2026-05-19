
import numpy as np



#  Para Gauss-Newton
def f_gn(x):
    return (x - 3)**2 + 2

def df_gn(x):
    return 2*(x - 3)

def d2f_gn(x):
    return 2


def Met_gauss_newton(funcion_derivada, funcion_segunda_derivada, x0, tol=1e-6, max_iter=50):
    x = x0

    for i in range(max_iter):
        x_new = x - funcion_derivada(x) / funcion_segunda_derivada(x)

        if abs(x_new - x) < tol:
            print(f"Convergió en {i} iteraciones")
            return x_new

        x = x_new

    return x

#  Para descenso de gradiente
def f_gd(x):
    return (x - 1)**(5/3)

def df_gd(x):
    return (5/3) * abs(x - 1)**(2/3)


def Met_descenso_gradiente(funcion_derivada, x0, gamma=0.3, tol=1e-6, max_iter=1000):
    x = x0

    for i in range(max_iter):
        x_new = x - gamma * funcion_derivada(x)

        if abs(x_new - x) < tol:
            print(f"Convergió en {i} iteraciones")
            return x_new

        x = x_new


    return x

#  Para secante
def f_sec(x):
    return np.sqrt(np.abs(x - 2))


def Met_secant_gradient(funcion, x1, x2, gamma=1, tol=1e-6, max_iter=1000):

    for i in range(max_iter):

        if abs(x2 - x1) < 1e-12:
            print("División por cero evitada")
            return x2

        grad_aprox = (funcion(x2) - funcion(x1)) / (x2 - x1)

        x3 = x2 - gamma * grad_aprox

        if abs(x3 - x2) < tol:
            print(f"Convergió en {i} iteraciones")
            return x3

        x1, x2 = x2, x3

    return x2




# --- Gauss-Newton ---
x0 = 0
print("Gauss-Newton:")
xmin_gn = Met_gauss_newton(df_gn, d2f_gn, x0)
print("Resultado:", xmin_gn)


# --- Descenso de gradiente ---
x0 = 4
print("\nDescenso de gradiente:")
xmin_gd = Met_descenso_gradiente(df_gd, x0, gamma=0.3)
print("Resultado:", xmin_gd)


# --- Secante ---
x1, x2 = 6, 4
print("\nMétodo de la secante:")
xmin_sec = Met_secant_gradient(f_sec, x1, x2, gamma=1)
print("Resultado:", xmin_sec)