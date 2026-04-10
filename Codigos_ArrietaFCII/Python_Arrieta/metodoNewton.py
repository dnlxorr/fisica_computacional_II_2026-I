import numpy as np
import math


# def funcion(x):
#     return math.cos(x) - x
#
#
# def derivada_funcion(x):
#     return -math.sin(x) - 1


def funcion(x):
    return x**3-x-2

#derivada de la funcion

def derivada_funcion(x):
    return 3*x**2-1


def metodo_newton(funcion, derivada_funcion, x0, tolerancia=1e-8, maxIter= 100):
    x= x0

    for i in range (maxIter):
        #evaluamos en la funcion y en la derivada de la funcion

        fx= funcion(x)
        dfx = derivada_funcion(x)

        if abs(dfx) < 1e-12:
            print("Error: la derivada de la funcion es 0 en el punto. El metodo no puede continuar")
            return None

        # formula metodo de Newton  x_nuevo = x - f(x)/f'(x)
        xNew= x - fx /dfx

        #calculo error
        error = abs(xNew - x)

        print(f"Iteracion {i+1}: x = {xNew:.6f},  f(x)= {funcion(xNew): .2f},  error = {error:.2e}")

        #criterio de parada si el error es menor a la tolerancia
        if error < tolerancia:
            print(f"solucion convergente, la raiz aproximada es: {xNew: .6f}")
            return xNew

        # actualizamos x para la siguiente iteracion
        x = xNew

    print("NO convergió en el rango de iteraciones dada")
    return None

print(" f(x) = x³ - x - 2\n")
raiz = metodo_newton(funcion, derivada_funcion, x0=2)


# print("\n>>> f(x) = cos(x) - x\n")
# raiz = metodo_newton(funcion, derivada_funcion, x0=1.0)

