import matplotlib.pyplot as plt
from numpy import zeros
from cmath import exp, pi,cos
from random import gauss

def dft(y):
    N = len(y)
    c = zeros(N//2 +1, complex)

    for k in range (N//2+1):
        for n in range (N):
            c[k] += y[n]*exp(-2j*pi*k*n/N)
    return c


# Generamos los datos del oscilador con ruido.

N= 1000

y = zeros(N,complex)
t = zeros(N)

A = 1
dt = 0.01
sigma = 1.0

for n in range(N):
    t[n] = n*dt
    señal = A*cos(2*pi*5*t[n]) + 0.7*cos(2*pi*12*t[n])
    ruido = gauss(0,sigma)
    y[n] = señal+ruido



plt.plot(t,y)
plt.xlabel('tiempo (s)')
plt.ylabel('Amplitud')
plt.title(f"Oscilacion con ruido blanco $\sigma$ = {sigma}")
plt.show()


c = dft(y)

#Eje de frecuencias

f = zeros(N//2+1)
for k in range(N//2+1):
    f[k] = k/(N*dt)

#Magnitud del espectro

C =zeros(N//2+1)
for k in range(N//2+1):
    C[k] = abs(c[k])/N


plt.plot(f,C)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel("Amplitud")
plt.xlim(0, 15)
plt.title("Espectro de Fourier")
plt.show()