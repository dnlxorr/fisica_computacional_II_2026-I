# Resolver f(x) = x^3 - x - 2 usando bisección

a = 1
b = 2
tolerancia = 1e-6
error = 1
iteracion = 0

while error > tolerancia:

    c = (a + b) / 2

    fa = a ** 3 - a - 2
    fc = c ** 3 - c - 2

    print("Iteración:", iteracion)
    print("a =", a, "b =", b, "c =", c)
    print("f(c) =", fc)
    print("-----------------------")

    if fa * fc < 0:
        b = c
    else:
        a = c

    error = abs(b - a)
    iteracion += 1

print("Raíz aproximada:", c)