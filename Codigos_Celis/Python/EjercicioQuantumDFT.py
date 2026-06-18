"""
════════════════════════════════════════════════════════════════════════════════
  EVOLUCIÓN TEMPORAL DE UN PAQUETE DE ONDAS GAUSSIANO — ANÁLISIS COMPLETO
  Física Computacional II  |  DFT O(N²) completa (corregida)
════════════════════════════════════════════════════════════════════════════════

CORRECCIONES respecto a la versión original:
  1. dft / idft — ahora manejan la grilla COMPLETA de N frecuencias (±k),
     no solo N//2+1. Esto es esencial para paquetes con k₀ ≠ 0.
  2. k_grid — devuelve las N frecuencias ordenadas como fftfreq:
     [0, 1, ..., N/2-1, -N/2, ..., -1] × 2π/(N·dx)
  3. sigma_numeric — corregido el factor √2:
     std(|ψ|²) = σ₀/√2  →  σ_ψ = √2 · std(|ψ|²)
  4. prob_analytic — factor de normalización corregido a 1/(√π·σ)
  5. energia_cinetica — adaptada a grilla completa (sin pesos hermíticos)
  6. norma — recibe psi en lugar de prob para consistencia
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 0 — Parámetros globales y funciones base
# ════════════════════════════════════════════════════════════════════════════

# ── Parámetros físicos (unidades naturales) ──────────────────────────────────
hbar   = 1.0
m      = 1.0
sigma0 = 1.0    # ancho inicial de referencia
k0     = 3.0    # momento inicial

# ── Parámetros numéricos ─────────────────────────────────────────────────────
N  = 128        # puntos en la grilla (potencia de 2)
L  = 20.0       # semiancho del dominio  x ∈ [-L, L]
dt = 0.05       # paso temporal
T  = 4.0        # tiempo total

# ── Grilla espacial ──────────────────────────────────────────────────────────
x  = np.linspace(-L, L, N, endpoint=False)
dx = x[1] - x[0]

# ── Compatibilidad NumPy < 2 y >= 2 ─────────────────────────────────────────
trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ────────────────────────────────────────────────────────────────────────────
#  DFT O(N²) COMPLETA — grilla bilateral (±k)
#
#  La versión original usaba solo k = 0..N//2, perdiendo las frecuencias
#  negativas. Un paquete con k₀=3 tiene su espectro centrado en k₀>0,
#  por lo que la reconstrucción con solo k≥0 era incorrecta.
#
#  Convención estándar (igual que numpy.fft):
#    φ[k] = Σ_{n=0}^{N-1} ψ[n] · exp(-2πi·k·n/N)      k = 0..N-1
#    ψ[n] = (1/N) Σ_{k=0}^{N-1} φ[k] · exp(+2πi·k·n/N) n = 0..N-1
#
#  Los índices k = N//2..N-1 corresponden a frecuencias negativas
#  k_phys = (k - N) × 2π/(N·dx), igual que fftfreq.
# ────────────────────────────────────────────────────────────────────────────

def dft(y):
    """
    DFT directa completa O(N²).
    Devuelve φ[k] para k = 0, 1, ..., N-1.
    Los índices k >= N//2 corresponden a frecuencias negativas.
    """
    N = len(y)
    c = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            c[k] += y[n] * np.exp(-2j * np.pi * k * n / N)
    return c


def idft(c):
    """
    IDFT completa O(N²) — inversa exacta de dft().
    ψ[n] = (1/N) Σ_k φ[k] · exp(+2πi·k·n/N)
    """
    N = len(c)
    y = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            y[n] += c[k] * np.exp(2j * np.pi * k * n / N)
        y[n] /= N
    return y


def k_grid(N, dx):
    """
    Grilla de momentos físicos compatible con dft() completa.
    Orden: [0, 1, ..., N/2-1, -N/2, ..., -1] × 2π/(N·dx)
    (mismo orden que numpy.fft.fftfreq × 2π)
    """
    k = np.zeros(N)
    for i in range(N):
        if i < N // 2:
            k[i] =  i
        else:
            k[i] = i - N
    return k * (2 * np.pi / (N * dx))


# ────────────────────────────────────────────────────────────────────────────
#  Funciones físicas
# ────────────────────────────────────────────────────────────────────────────

def psi0_gauss(x, sigma0, k0):
    """Estado inicial gaussiano normalizado."""
    norm = (sigma0**2 * np.pi) ** (-0.25)
    return norm * np.exp(-x**2 / (2 * sigma0**2)) * np.exp(1j * k0 * x)


def evolve_k(phi_k, k_arr, dt):
    """Operador de evolución cinética en espacio k (grilla completa)."""
    return phi_k * np.exp(-1j * hbar * k_arr**2 / (2 * m) * dt)


def sigma_t_analytic(sigma0, t):
    """Ancho analítico exacto σ_ψ(t) del gaussiano en la amplitud ψ."""
    return sigma0 * np.sqrt(1 + (hbar * t / (m * sigma0**2))**2)


def prob_analytic(x, sigma0, k0, t):
    """
    Densidad de probabilidad analítica |ψ(x,t)|².
    ψ ~ exp(-x²/2σ²)  →  |ψ|² ~ exp(-x²/σ²) / (√π · σ).
    CORRECCIÓN: normalizacion 1/(√π·σ), no 1/(√(2π)·σ).
    """
    sig     = sigma_t_analytic(sigma0, t)
    x_cen   = hbar * k0 / m * t
    norm_sq = (sigma0**2 * np.pi) ** (-0.5)   # |norm|² de psi0
    return norm_sq * (sigma0 / sig) * np.exp(-(x - x_cen)**2 / sig**2)


def sigma_numeric(x, psi):
    """
    Desviación estándar σ_ψ del gaussiano en la AMPLITUD ψ.

    Como ψ ~ exp(-x²/2σ_ψ²), la densidad es |ψ|² ~ exp(-x²/σ_ψ²),
    cuya varianza estadística es σ_ψ²/2. Por tanto:
        std(|ψ|²) = σ_ψ/√2   →   σ_ψ = √2 · std(|ψ|²)

    CORRECCIÓN: se aplica el factor √2 que faltaba en la versión original.
    """
    prob = np.abs(psi)**2
    norm = trapz(prob, x)
    mu   = trapz(x * prob, x) / norm
    var  = trapz((x - mu)**2 * prob, x) / norm
    return np.sqrt(2 * var)   # ← factor √2 corregido


def x_mean(x, psi):
    """Valor esperado ⟨x⟩."""
    prob = np.abs(psi)**2
    norm = trapz(prob, x)
    return trapz(x * prob, x) / norm


def norma(psi, x):
    """Norma ∫|ψ|²dx (debe conservarse ≈ 1)."""
    return trapz(np.abs(psi)**2, x)


def corriente_prob(psi, x):
    """Corriente de probabilidad J(x,t) = (ℏ/m) Im[ψ* dψ/dx]."""
    dpsi_dx = np.gradient(psi, x)
    return (hbar / m) * np.imag(np.conj(psi) * dpsi_dx)


def energia_cinetica(phi_k, k_arr, dx):
    ek = hbar**2 * k_arr**2 / (2 * m)
    return (dx / N) * np.sum(ek * np.abs(phi_k)**2)


# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 1 — Simulación principal con sigma0 de referencia
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Simulación principal")
print(f"  N={N}, L={L}, σ₀={sigma0}, k₀={k0}, dt={dt}, T={T}")
print("  DFT O(N²) completa (±k) — puede tardar ~20-60 s con N=128")
print("=" * 60)

k_arr   = k_grid(N, dx)
n_steps = int(T / dt)

psi   = psi0_gauss(x, sigma0, k0)
phi_k = dft(psi)

# Almacenamiento
times      = [0.0]
probs      = [np.abs(psi)**2]
phi_ks     = [phi_k.copy()]
sigmas_num = [sigma_numeric(x, psi)]
sigmas_an  = [sigma_t_analytic(sigma0, 0.0)]
x_means    = [x_mean(x, psi)]
normas     = [norma(psi, x)]
energias   = [energia_cinetica(phi_k, k_arr,dx)]
corrientes = [corriente_prob(psi, x)]
fases      = [np.angle(psi)]
psis       = [psi.copy()]

for step in range(1, n_steps + 1):
    phi_k = evolve_k(phi_k, k_arr, dt)
    psi   = idft(phi_k)
    t_now = step * dt

    times.append(t_now)
    probs.append(np.abs(psi)**2)
    phi_ks.append(phi_k.copy())
    sigmas_num.append(sigma_numeric(x, psi))
    sigmas_an.append(sigma_t_analytic(sigma0, t_now))
    x_means.append(x_mean(x, psi))
    normas.append(norma(psi, x))
    energias.append(energia_cinetica(phi_k, k_arr,dx))
    corrientes.append(corriente_prob(psi, x))
    fases.append(np.angle(psi))
    psis.append(psi.copy())

times      = np.array(times)
sigmas_num = np.array(sigmas_num)
sigmas_an  = np.array(sigmas_an)
x_means    = np.array(x_means)
normas     = np.array(normas)
energias   = np.array(energias)

print("  ✓ Simulación completa\n")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2 — Figura 1: Multipanel densidad |ψ(x,t)|²
# ════════════════════════════════════════════════════════════════════════════
t_snap = [0.0, T/4, T/2, 3*T/4, T]
fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8))
fig1.suptitle(
    f"Evolución de |ψ(x,t)|² — DFT O(N²) completa\n"
    f"σ₀={sigma0}, k₀={k0}, N={N}, L={L}",
    fontsize=13, fontweight="bold"
)

cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(t_snap)))

for ax, ts, col in zip(axes1.flat, t_snap, cmap):
    idx    = min(range(len(times)), key=lambda i: abs(times[i] - ts))
    prob_n = probs[idx]
    prob_a = prob_analytic(x, sigma0, k0, times[idx])
    ax.fill_between(x, prob_n, alpha=0.35, color=col)
    ax.plot(x, prob_n, color=col,  lw=2,   label="DFT numérico")
    ax.plot(x, prob_a, "k--",      lw=1.3, label="Analítico")
    ax.set_title(f"t = {times[idx]:.2f}  |  σ = {sigmas_num[idx]:.3f}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("|ψ|²")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

# Panel 6: σ(t)
ax_s = axes1[1, 2]
ax_s.plot(times, sigmas_num, "b-",  lw=2,   label="σ(t) numérico")
ax_s.plot(times, sigmas_an,  "r--", lw=1.5,
          label=r"$\sigma_0\sqrt{1+(\hbar t/m\sigma_0^2)^2}$")
ax_s.fill_between(times, sigmas_num, sigmas_an, alpha=0.2,
                   color="orange", label="Error")
ax_s.set_xlabel("t")
ax_s.set_ylabel("σ(t)")
ax_s.set_title("Dispersión del paquete", fontsize=10)
ax_s.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig1_multipanel_densidad.png", dpi=150, bbox_inches="tight")
print("  ✓ fig1_multipanel_densidad.png")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 3 — Figura 2: Espectro de momentos |φ(k,t)|²
# ════════════════════════════════════════════════════════════════════════════
# Para visualizar, reordenamos el espectro de [-k_max, k_max)
def fftshift_manual(arr, k_arr):
    """Reordena arr y k_arr de [0..kmax, -kmax..-1] a [-kmax..kmax]."""
    mid  = N // 2
    arr_s = np.concatenate([arr[mid:], arr[:mid]])
    k_s   = np.concatenate([k_arr[mid:], k_arr[:mid]])
    return arr_s, k_s

k_plot, _ = fftshift_manual(np.abs(phi_ks[0])**2, k_arr)
_, k_plot_vals = fftshift_manual(np.abs(phi_ks[0])**2, k_arr)
k_sorted = np.concatenate([k_arr[N//2:], k_arr[:N//2]])

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Espectro de momentos |φ(k,t)|²", fontsize=13, fontweight="bold")

t_ids = [0, n_steps//4, n_steps//2, n_steps]
cmap2 = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_ids)))

for idx, col in zip(t_ids, cmap2):
    spec_raw = np.abs(phi_ks[idx])**2
    spec_s   = np.concatenate([spec_raw[N//2:], spec_raw[:N//2]])
    axes2[0].plot(k_sorted, spec_s, color=col, lw=1.8,
                  label=f"t={times[idx]:.2f}")

axes2[0].set_xlabel("k")
axes2[0].set_ylabel("|φ(k,t)|²")
axes2[0].set_title("Espectro en distintos tiempos\n(debe ser constante — V=0)")
axes2[0].legend(fontsize=9)
axes2[0].set_xlim(-2*k0, 2*k0)

# Variación máxima del espectro
spec0   = np.abs(phi_ks[0])**2
max_var = [np.max(np.abs(np.abs(phi_ks[i])**2 - spec0)) for i in range(len(phi_ks))]
axes2[1].semilogy(times, np.array(max_var) + 1e-16, "g-", lw=2)
axes2[1].set_xlabel("t")
axes2[1].set_ylabel("max|Δ|φ|²|  (escala log)")
axes2[1].set_title("Variación máxima del espectro\n(verifica unitaridad del propagador)")
axes2[1].grid(True, which="both", ls=":", alpha=0.5)

plt.tight_layout()
plt.savefig("fig2_espectro_momentos.png", dpi=150, bbox_inches="tight")
print("  ✓ fig2_espectro_momentos.png")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 4 — Figura 3: Observables temporales
# ════════════════════════════════════════════════════════════════════════════
x_class     = hbar * k0 / m * times
E_analitica = hbar**2 / (2*m) * (k0**2 + 1/(2*sigma0**2))

fig3, axes3 = plt.subplots(2, 2, figsize=(13, 8))
fig3.suptitle("Observables temporales", fontsize=13, fontweight="bold")

# ⟨x⟩(t)
axes3[0, 0].plot(times, x_means, "b-",  lw=2,   label="⟨x⟩ numérico")
axes3[0, 0].plot(times, x_class, "r--", lw=1.5, label="ℏk₀t/m (clásico)")
axes3[0, 0].set_xlabel("t")
axes3[0, 0].set_ylabel("⟨x⟩(t)")
axes3[0, 0].set_title("Posición media — Teorema de Ehrenfest")
axes3[0, 0].legend(fontsize=9)

# Norma
axes3[0, 1].plot(times, normas, "m-", lw=2, label="∫|ψ|²dx")
axes3[0, 1].axhline(1.0, color="k", ls="--", lw=1, label="Norma exacta = 1")
axes3[0, 1].set_xlabel("t")
axes3[0, 1].set_ylabel("Norma")
axes3[0, 1].set_title("Conservación de probabilidad")
axes3[0, 1].set_ylim(0.98, 1.02)
axes3[0, 1].legend(fontsize=9)

# Energía
axes3[1, 0].plot(times, energias, color="orange", lw=2, label="⟨E⟩ numérico")
axes3[1, 0].axhline(E_analitica, color="k", ls="--", lw=1,
                     label=f"⟨E⟩ analítico = {E_analitica:.4f}")
axes3[1, 0].set_xlabel("t")
axes3[1, 0].set_ylabel("⟨E⟩")
axes3[1, 0].set_title("Energía cinética media")
axes3[1, 0].legend(fontsize=9)

# Error relativo en σ(t)
err_rel = np.abs(sigmas_num - sigmas_an) / sigmas_an * 100
axes3[1, 1].semilogy(times, err_rel + 1e-12, "r-", lw=2)
axes3[1, 1].set_xlabel("t")
axes3[1, 1].set_ylabel("Error relativo σ(t) [%]")
axes3[1, 1].set_title("Precisión numérica de σ(t)")
axes3[1, 1].grid(True, which="both", ls=":", alpha=0.5)

plt.tight_layout()
plt.savefig("fig3_observables.png", dpi=150, bbox_inches="tight")
print("  ✓ fig3_observables.png")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 5 — Figura 4: Corriente de probabilidad J(x,t)
# ════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))
fig4.suptitle("Corriente de probabilidad J(x,t) = (ℏ/m) Im[ψ* ∂ψ/∂x]",
              fontsize=13, fontweight="bold")

cmap4 = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(t_snap)))
for ts, col in zip(t_snap, cmap4):
    idx = min(range(len(times)), key=lambda i: abs(times[i] - ts))
    axes4[0].plot(x, corrientes[idx], color=col, lw=1.8,
                  label=f"t={times[idx]:.2f}")

axes4[0].axhline(0, color="k", lw=0.7, ls=":")
axes4[0].set_xlabel("x")
axes4[0].set_ylabel("J(x,t)")
axes4[0].set_title("Corriente en varios tiempos")
axes4[0].legend(fontsize=8)

# Ecuación de continuidad
idx_mid = n_steps // 2
dpdt    = (np.array(probs[idx_mid+1]) - np.array(probs[idx_mid-1])) / (2*dt)
dJdx    = np.gradient(corrientes[idx_mid], x)
resid   = dpdt + dJdx

axes4[1].plot(x, dpdt,  "b-",  lw=1.5, label="∂|ψ|²/∂t")
axes4[1].plot(x, -dJdx, "r--", lw=1.5, label="-∂J/∂x")
axes4[1].plot(x, resid, "g:",  lw=1,   label="Residuo (debe ≈ 0)")
axes4[1].set_xlabel("x")
axes4[1].set_title(f"Ecuación de continuidad en t={times[idx_mid]:.2f}")
axes4[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig4_corriente.png", dpi=150, bbox_inches="tight")
print("  ✓ fig4_corriente.png")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 6 — Figura 5: Principio de incertidumbre Δx·Δp(t)
# ════════════════════════════════════════════════════════════════════════════
Delta_p     = hbar / (2 * sigma0)
Delta_x     = sigmas_num
producto    = Delta_x * Delta_p
producto_an = sigmas_an * Delta_p

fig5, ax5 = plt.subplots(figsize=(9, 5))
ax5.plot(times, producto,    "b-",  lw=2.5, label="Δx(t)·Δp  numérico")
ax5.plot(times, producto_an, "r--", lw=1.5, label=r"Δx(t)·Δp  analítico")
ax5.axhline(hbar/2, color="k", ls=":", lw=1.5, label=r"$\hbar/2$ (mínimo)")
ax5.fill_between(times, hbar/2, producto, alpha=0.15, color="blue",
                  label="Exceso sobre mínimo")
ax5.set_xlabel("t", fontsize=12)
ax5.set_ylabel(r"$\Delta x \cdot \Delta p$", fontsize=12)
ax5.set_title(
    "Principio de Incertidumbre de Heisenberg\n"
    rf"$\Delta x(t)\cdot\Delta p \geq \hbar/2$  —  σ₀={sigma0}, k₀={k0}",
    fontsize=12
)
ax5.legend(fontsize=9)
ax5.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("fig5_incertidumbre.png", dpi=150, bbox_inches="tight")
print("  ✓ fig5_incertidumbre.png")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 7 — Figura 6: Fase de ψ (chirp cuántico)
# ════════════════════════════════════════════════════════════════════════════
fig6, axes6 = plt.subplots(1, 2, figsize=(13, 5))
fig6.suptitle("Fase de ψ(x,t) — Chirp cuántico", fontsize=13, fontweight="bold")

t_ids_fase = [0, n_steps//4, n_steps//2, n_steps]
cmap6 = plt.cm.magma(np.linspace(0.1, 0.9, len(t_ids_fase)))

for idx, col in zip(t_ids_fase, cmap6):
    prob_mask = probs[idx] > probs[idx].max() * 0.02
    fase_plot = np.where(prob_mask, fases[idx], np.nan)
    axes6[0].plot(x, fase_plot, color=col, lw=1.8, label=f"t={times[idx]:.2f}")

axes6[0].set_xlabel("x")
axes6[0].set_ylabel("arg(ψ) [rad]")
axes6[0].set_title("Fase en la región del paquete")
axes6[0].legend(fontsize=9)

fase_matrix = np.array(fases)
im = axes6[1].pcolormesh(x, times, fase_matrix, cmap="RdBu", shading="auto")
plt.colorbar(im, ax=axes6[1], label="arg(ψ) [rad]")
axes6[1].set_xlabel("x")
axes6[1].set_ylabel("t")
axes6[1].set_title("Mapa de fase ψ(x,t)")

plt.tight_layout()
plt.savefig("fig6_fase_chirp.png", dpi=150, bbox_inches="tight")
print("  ✓ fig6_fase_chirp.png")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 8 — Figura 7: Comparación σ(t) para múltiples σ₀
# ════════════════════════════════════════════════════════════════════════════
print("\n  Calculando variación de σ₀ (puede tardar) ...")

sigmas_0_lista = [0.25, 0.5, 1.0, 2.0, 4.0]
cmap7          = plt.cm.tab10(np.linspace(0, 0.8, len(sigmas_0_lista)))

resultados_sigma0 = {}

for s0 in sigmas_0_lista:
    psi_s    = psi0_gauss(x, s0, k0)
    phi_k_s  = dft(psi_s)
    sig_t_s  = [sigma_numeric(x, psi_s)]
    sig_an_s = [sigma_t_analytic(s0, 0.0)]

    for step in range(1, n_steps + 1):
        phi_k_s = evolve_k(phi_k_s, k_arr, dt)
        psi_s   = idft(phi_k_s)
        sig_t_s.append(sigma_numeric(x, psi_s))
        sig_an_s.append(sigma_t_analytic(s0, step * dt))

    resultados_sigma0[s0] = (np.array(sig_t_s), np.array(sig_an_s))
    print(f"    σ₀ = {s0:.2f}  ✓")

fig7, axes7 = plt.subplots(1, 2, figsize=(14, 6))
fig7.suptitle("Efecto de σ₀ sobre la dispersión del paquete",
              fontsize=13, fontweight="bold")

for s0, col in zip(sigmas_0_lista, cmap7):
    sig_n, sig_a = resultados_sigma0[s0]
    tau = m * s0**2 / hbar
    axes7[0].plot(times, sig_n, color=col, lw=2,
                  label=f"σ₀={s0} (τ={tau:.2f})")
    axes7[0].plot(times, sig_a, color=col, lw=1, ls="--", alpha=0.6)

for s0, col in zip(sigmas_0_lista, cmap7):
    tau = m * s0**2 / hbar
    if tau <= T:
        axes7[0].axvline(tau, color=col, lw=0.8, ls=":", alpha=0.7)

axes7[0].set_xlabel("t")
axes7[0].set_ylabel("σ(t)")
axes7[0].set_title("σ(t) numérico (sólido) y analítico (punteado)\n"
                    "Líneas verticales: τ = mσ₀²/ℏ")
axes7[0].legend(fontsize=8)

t_asy = times[times > 0.5]
for s0, col in zip(sigmas_0_lista, cmap7):
    asy = hbar * t_asy / (m * s0)
    axes7[1].plot(t_asy, asy, color=col, lw=2,
                  label=f"σ₀={s0}: σ≈ℏt/mσ₀")

axes7[1].set_xlabel("t")
axes7[1].set_ylabel("σ asintótico ≈ ℏt/mσ₀")
axes7[1].set_title("Comportamiento asintótico t >> τ\n"
                    "σ(t) ≈ ℏt/(mσ₀) — lineal en t")
axes7[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig7_comparacion_sigma0.png", dpi=150, bbox_inches="tight")
print("  ✓ fig7_comparacion_sigma0.png")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 9 — Figura 8: Producto Δx·Δp para múltiples σ₀
# ════════════════════════════════════════════════════════════════════════════
fig8, ax8 = plt.subplots(figsize=(10, 6))
ax8.axhline(hbar/2, color="k", ls=":", lw=2,
             label=r"Mínimo $\hbar/2$", zorder=5)

for s0, col in zip(sigmas_0_lista, cmap7):
    sig_n, _ = resultados_sigma0[s0]
    dp   = hbar / (2 * s0)
    prod = sig_n * dp
    ax8.plot(times, prod, color=col, lw=2,
              label=f"σ₀={s0},  Δp={dp:.3f}")

ax8.set_xlabel("t", fontsize=12)
ax8.set_ylabel(r"$\Delta x(t) \cdot \Delta p$", fontsize=12)
ax8.set_title(
    "Producto de incertidumbres Δx·Δp para distintos σ₀\n"
    r"Todos parten de $\hbar/2$ en t=0 y crecen para t>0",
    fontsize=12
)
ax8.legend(fontsize=9)
ax8.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("fig8_incertidumbre_multi_sigma.png", dpi=150, bbox_inches="tight")
print("  ✓ fig8_incertidumbre_multi_sigma.png")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 10 — Figura 9: Tiempo característico τ vs σ₀
# ════════════════════════════════════════════════════════════════════════════
s0_arr  = np.linspace(0.1, 5.0, 200)
tau_arr = m * s0_arr**2 / hbar
dk_arr  = 1 / (2 * s0_arr)

fig9, axes9 = plt.subplots(1, 2, figsize=(13, 5))
fig9.suptitle("Parámetros de dispersión en función de σ₀",
              fontsize=13, fontweight="bold")

axes9[0].plot(s0_arr, tau_arr, "b-", lw=2.5,
               label=r"$\tau = m\sigma_0^2/\hbar$")
axes9[0].scatter(sigmas_0_lista, [m*s**2/hbar for s in sigmas_0_lista],
                  color="red", zorder=5, s=60, label="Valores simulados")
axes9[0].set_xlabel("σ₀")
axes9[0].set_ylabel("τ")
axes9[0].set_title(r"Tiempo característico $\tau \propto \sigma_0^2$")
axes9[0].legend(fontsize=9)
axes9[0].grid(True, ls=":", alpha=0.4)

axes9[1].plot(s0_arr, dk_arr, "r-", lw=2.5,
               label=r"$\Delta k = 1/(2\sigma_0)$")
axes9[1].scatter(sigmas_0_lista, [1/(2*s) for s in sigmas_0_lista],
                  color="blue", zorder=5, s=60, label="Valores simulados")
ax9r = axes9[1].twinx()
ax9r.plot(s0_arr, hbar**2 / (2*m*s0_arr**2), "g--", lw=1.5,
           label="Energía de fluct. ℏ²/2mσ₀²")
ax9r.set_ylabel("Energía de fluctuación", color="g")
axes9[1].set_xlabel("σ₀")
axes9[1].set_ylabel("Δk")
axes9[1].set_title(r"Ancho espectral $\Delta k = 1/(2\sigma_0)$")
axes9[1].legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig("fig9_tau_vs_sigma0.png", dpi=150, bbox_inches="tight")
print("  ✓ fig9_tau_vs_sigma0.png")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 11 — Animación GIF
# ════════════════════════════════════════════════════════════════════════════
print("\n  Generando animación GIF ...")

fig_ani, ax_ani = plt.subplots(figsize=(9, 5))
ax_ani.set_xlim(-L, L)
y_max = max(p.max() for p in probs) * 1.18
ax_ani.set_ylim(0, y_max)
ax_ani.set_xlabel("x", fontsize=12)
ax_ani.set_ylabel("|ψ(x,t)|²", fontsize=12)
ax_ani.set_title("Evolución del paquete de ondas gaussiano (DFT)", fontsize=12)

line_num, = ax_ani.plot(x, probs[0], "b-",  lw=2,   label="DFT numérico")
line_an,  = ax_ani.plot(x, prob_analytic(x, sigma0, k0, 0), "r--",
                          lw=1.5, label="Analítico")
fill_col  = [ax_ani.fill_between(x, probs[0], alpha=0.25, color="steelblue")]
time_txt  = ax_ani.text(0.02, 0.93, "t = 0.00",
                         transform=ax_ani.transAxes, fontsize=11)
sig_txt   = ax_ani.text(0.02, 0.85, f"σ = {sigma0:.3f}",
                         transform=ax_ani.transAxes, fontsize=10, color="gray")
ax_ani.legend(loc="upper right", fontsize=9)


def init_anim():
    line_num.set_data(x, probs[0])
    line_an.set_data(x, prob_analytic(x, sigma0, k0, 0))
    return line_num, line_an, time_txt, sig_txt


def update_anim(frame):
    pn = probs[frame]
    pa = prob_analytic(x, sigma0, k0, times[frame])
    line_num.set_data(x, pn)
    line_an.set_data(x, pa)
    fill_col[0].remove()
    fill_col[0] = ax_ani.fill_between(x, pn, alpha=0.25, color="steelblue")
    time_txt.set_text(f"t = {times[frame]:.2f}")
    sig_txt.set_text(f"σ = {sigmas_num[frame]:.3f}")
    return line_num, line_an, time_txt, sig_txt


ani = animation.FuncAnimation(
    fig_ani, update_anim, frames=len(times),
    init_func=init_anim, interval=50, blit=False
)
ani.save("wavepacket_animation.gif", writer="pillow", fps=20)
print("  ✓ wavepacket_animation.gif")

# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 12 — Tabla comparativa en consola
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  TABLA: Comparación σ(t) numérico vs analítico")
print("═"*65)
print(f"  σ₀={sigma0}, k₀={k0}, N={N}, L={L}, dt={dt}")
print(f"  {'t':>6}  {'σ numérico':>12}  {'σ analítico':>12}  {'error %':>9}  {'norma':>8}")
print("  " + "-"*60)
indices = range(0, len(times), max(1, len(times)//12))
for i in indices:
    err = abs(sigmas_num[i] - sigmas_an[i]) / sigmas_an[i] * 100
    print(f"  {times[i]:6.2f}  {sigmas_num[i]:12.6f}  {sigmas_an[i]:12.6f}"
          f"  {err:9.5f}%  {normas[i]:8.6f}")

print("\n" + "═"*65)
print("  TABLA: Efecto de σ₀ — parámetros clave")
print("═"*65)
print(f"  {'σ₀':>6}  {'τ=mσ₀²/ℏ':>10}  {'Δk=1/2σ₀':>10}  "
      f"{'σ(t=4) num':>12}  {'σ(t=4) an':>12}  {'error %':>9}")
print("  " + "-"*65)
for s0 in sigmas_0_lista:
    tau = m * s0**2 / hbar
    dk  = 1 / (2 * s0)
    sn  = resultados_sigma0[s0][0][-1]
    sa  = resultados_sigma0[s0][1][-1]
    err = abs(sn - sa) / sa * 100
    print(f"  {s0:6.2f}  {tau:10.4f}  {dk:10.4f}  {sn:12.6f}  {sa:12.6f}  {err:9.5f}%")

print("\n  Archivos generados:")
archivos = [
    "fig1_multipanel_densidad.png",
    "fig2_espectro_momentos.png",
    "fig3_observables.png",
    "fig4_corriente.png",
    "fig5_incertidumbre.png",
    "fig6_fase_chirp.png",
    "fig7_comparacion_sigma0.png",
    "fig8_incertidumbre_multi_sigma.png",
    "fig9_tau_vs_sigma0.png",
    "wavepacket_animation.gif",
]
for a in archivos:
    print(f"    ✓  {a}")

plt.show()

