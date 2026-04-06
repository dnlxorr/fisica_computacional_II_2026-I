import math
from math import exp



def f(x):
    return math.sqrt(1-math.log(x))
    #2 - exp(-x) #exp(1-x*x) #math.sqrt(1-math.log(x)) #x*x+math.sin(2x) #0.5 * math.asin(x - x**2)

# Derivada numérica (diferencias finitas)

def derivada(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)



# Metodo de relajacion general

def metodoRelajacion(f, xini, tolerancia=1e-6, max_iter=100):
    x = xini

    for i in range(max_iter):
        x_new = f(x)
        print(f"Iter {i}: x = {x_new}")

        # Evaluar derivada en el punto actual
        d = derivada(f, x)

        # Verificación de convergencia teórica
        if abs(d) >= 1:
            print("Error: El metodo puede no converger (|f'(x)| >= 1).")
            return None

        # Criterio de convergencia
        if abs(x_new - x) < tolerancia:
            return x_new, i

        x = x_new

    print("Error: No convergió en el número máximo de iteraciones.")
    return None

resultado = metodoRelajacion(f, xini=1.0)

if resultado is not None:
    solucion, iteraciones = resultado
    print("Solución:", solucion)
    print("Iteraciones:", iteraciones)