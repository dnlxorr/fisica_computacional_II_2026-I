from math import sin

from matplotlib.pyplot import title
from numpy import arange
from pylab import plot,xlabel,ylabel,show


def f(x,t):
    return -x**3 + sin(t)

a = 0.0
b = 10.0
N = 50
h = (b-a)/N

tpoints = arange(a,b,h)
xpoints = []

x = 0.0
for t in tpoints:
    xpoints.append(x)
    k1= h*f(x,t)
    k2= h*f(x+0.5*k1,t)
    x += k2


plot(tpoints,xpoints)
title('Runge-Kutta 2O')
xlabel('t')
ylabel('x(t)')
show()