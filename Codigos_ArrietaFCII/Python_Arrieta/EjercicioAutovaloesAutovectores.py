"""
Ejercicio 6.9 - Pozo cuántico asimétrico
Mark Newman - Computational Physics

Electrón en un pozo de potencial 1D con V(x) = a*x/L.
Se resuelve la ecuación de Schrödinger como problema de valores propios
matricial, expandiendo ψ(x) en una base de senos de Fourier.

Partes:
  b) Expresión analítica de Hmn y construcción de la matriz
  c) Matriz 10x10  -> primeros 10 niveles de energía
  d) Matriz 100x100 -> comparación de convergencia
  e) Funciones de onda de los 3 primeros estados
"""

import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt


# Constantes físicas

hbar  = 1.0546e-34    # J·s
M     = 9.1094e-31    # kg  (masa del electrón)
eV    = 1.6022e-19    # J   (1 electrón-volt)
L     = 5e-10         # m   (5 Å, ancho del pozo)
a_pot = 10 * eV       # J   (altura del potencial, 10 eV)


def H_mn(m, n):
    """
    Elemento matricial del Hamiltoniano para V(x) = a*x/L.

    H_mn = (2/L) * integral_0^L  sin(m*pi*x/L) * H_op * sin(n*pi*x/L) dx

    Parte cinética (diagonal):
        delta_mn * hbar^2 * pi^2 * n^2 / (2 * M * L^2)

    Parte potencial — integral analítica (del libro, pág. 249):
        I(m,n) = integral_0^L  x * sin(m*pi*x/L) * sin(n*pi*x/L) dx
          m == n              ->  L^2 / 4
          m != n, misma paridad ->  0
          m != n, paridad dist. ->  -2*L^2*m*n / (pi^2 * (m^2-n^2)^2)

        Contribución: (2/L) * (a/L) * I(m,n)
    """
    # Parte cinética
    kinetic = (hbar**2 * np.pi**2 * n**2) / (2 * M * L**2) if m == n else 0.0

    # Parte potencial
    if m == n:
        I = L**2 / 4.0
    elif (m % 2) == (n % 2):   # misma paridad
        I = 0.0
    else:                       # paridad distinta
        I = -2.0 * L**2 * m * n / (np.pi**2 * (m**2 - n**2)**2)

    potential = (2.0 / L) * (a_pot / L) * I

    return kinetic + potential


def construir_H(size):
    """Construye la matriz hamiltoniana de dimensión size x size."""
    H = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            H[i, j] = H_mn(i + 1, j + 1)   # índices 1-based en la fórmula
    return H



# Parte c) Matriz 10x10

H10 = construir_H(10)
print(f"Matriz simétrica: {np.allclose(H10, H10.T)}")

evals10, evecs10 = np.linalg.eigh(H10)

print("\n" + "=" * 45)
print("Parte c) H 10×10 — primeros 10 niveles de energía")
print("=" * 45)
for k, e in enumerate(evals10):
    print(f"  E{k+1:2d} = {e / eV:8.4f} eV")


# Parte d) Matriz 100x100

H100 = construir_H(100)
evals100, evecs100 = np.linalg.eigh(H100)

print("\n" + "=" * 45)
print("Parte d) H 100×100 — primeros 10 niveles")
print("=" * 45)
for k in range(10):
    diff = abs(evals100[k] - evals10[k]) / eV
    print(f"  E{k+1:2d} = {evals100[k] / eV:8.4f} eV   (Δ vs 10×10 = {diff:.4f} eV)")

print("\nConclusión: los niveles convergen muy rápido;")
print("la diferencia entre 10×10 y 100×100 es < 0.001 eV para los primeros estados.")


# Parte e) Funciones de onda de los 3 primeros estados

x = np.linspace(0, L, 500)
Nterms = 100          # usamos los 100 coeficientes de la base

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
titles = [
    f"Estado base  E₁ = {evals100[0]/eV:.3f} eV",
    f"1er excitado  E₂ = {evals100[1]/eV:.3f} eV",
    f"2do excitado  E₃ = {evals100[2]/eV:.3f} eV",
]

for state in range(3):
    coeffs = evecs100[:, state]   # vector propio (Nterms,)

    # Reconstruir ψ(x) = Σ_n c_n * sin(n*π*x/L)
    psi = np.zeros(len(x))
    for n_idx in range(Nterms):
        n = n_idx + 1
        psi += coeffs[n_idx] * np.sin(n * np.pi * x / L)

    # Normalizar: ∫|ψ|² dx = 1
    norm = trapezoid(psi**2, x)
    psi /= np.sqrt(norm)

    prob = psi**2
    check = trapezoid(prob, x)

    axes[state].plot(x * 1e10, prob, linewidth=2, color='lightcoral')
    axes[state].fill_between(x * 1e10, prob, alpha=0.15, color='grey')
    axes[state].set_xlabel("x (Å)")
    axes[state].set_ylabel("|ψ(x)|²  (Å⁻¹)" if state == 0 else "")
    axes[state].set_title(titles[state], fontsize=11)
    axes[state].grid(True, alpha=0.3)
    axes[state].text(0.97, 0.95, f"∫|ψ|²dx = {check:.4f}",
                     transform=axes[state].transAxes,
                     ha='right', va='top', fontsize=9,
                     color='gray')

plt.suptitle("Ejercicio 6.9 e)  |ψ(x)|² para los primeros 3 estados", fontsize=12)
plt.tight_layout()
plt.savefig("ejercicio_6_9.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfica guardada como ejercicio_6_9.png")
