from math import sin
from numpy import arange
from pylab import plot,xlabel,ylabel,show

def f(x,t):
    return -x**3 + sin(t)

a = 0.0 # Comienzo del intervalo
b = 30.0 # FIn del intervalo
N = 1000 # NUmero de pasos
h = (b-a)/N # Tamaño del paso
x = 0.0 # Condición incial


tpoints = arange(a,b,h)
xpoints = []

for t in tpoints:
    xpoints.append(x)
    x += h*f(x,t)

plot(tpoints, xpoints)
xlabel("t")
ylabel("x")
show()

