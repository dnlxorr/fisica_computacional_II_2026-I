from math import exp, sqrt

sigma = 1.0   # VAlor de sigma en nm
tolerancia = 1e-6
z = (1+sqrt(5))/2 # Proporción aurea

def f(r):
    return (sigma/r)**6 - exp(-r/sigma)


# posiciones iniciales de los 4 puntos

x1 = sigma/10
x4 = sigma*10
x2 = x4 - (x4-x1)/z
x3 = x1 + (x4-x1)/z

# valores inciales de la funcion en los 4 puntos

f1= f(x1)
f2= f(x2)
f3= f(x3)
f4= f(x4)

while x4-x1 > tolerancia:
    if f2<f3:
        x4,f4 = x3,f3
        x3,f3 = x2,f2
        x2=x4-(x4-x1)/z
        f2 = f(x2)
    else:
        x1,f1 = x2,f2
        x2,f2 = x3,f3
        x3 = x1 + (x4-x1)/z
        f3 = f(x3)

print("El minímo está en: ",(x1+x4)/2," nm")