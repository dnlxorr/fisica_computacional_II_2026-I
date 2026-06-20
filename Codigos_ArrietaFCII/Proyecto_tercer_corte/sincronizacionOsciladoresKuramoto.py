"""
================================================================
 MODELO DE KURAMOTO — RESULTADOS COMPLETOS DEL PROYECTO
 Estudio numérico de la transición de sincronización y
 analogías con ritmos neuronales

 Figuras estáticas producidas:
   fig1_r_vs_K.png             — <r> vs K para 3 distribuciones
   fig2_r_temporal.png         — r(t) bajo/crítico/alto
   fig3_circular_snapshots.png — snapshots en S^1
   fig4_freq_efectivas.png     — Ω_i vs ω_i
   fig5_fft_espectro.png       — FFT de r(t)
   fig6_validacion.png         — numérico vs analítico + residuos
   fig7_neural_bandas.png      — analogía con bandas EEG
   fig8_panel_neural.png       — panel completo neurociencia
   fig9_resumen.png            — panel resumen del artículo

 Animaciones producidas:
   anim1_sincronizacion.gif    — fases en S^1 evolucionando en K
   anim2_transicion_rt.gif     — r(t) animado para K creciente
   anim3_neural_eeg.gif        — señal EEG simulada + r(t)

 Autor : Andrés E. Arrieta Lozano
 Curso : Física Computacional — U. de Pamplona
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
import warnings

warnings.filterwarnings("ignore")

# ── Reproducibilidad ─────────────────────────────────────────
SEED = 42
RNG = np.random.default_rng(SEED)

# ================================================================
# 0.  CONFIGURACIÓN GLOBAL
# ================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 130,
    "lines.linewidth": 1.8,
})

COL = {
    "lor": "#1f77b4",
    "gau": "#d62728",
    "uni": "#2ca02c",
    "anal": "#ff7f0e",
    "Kc": "#7f7f7f",
    # bandas EEG
    "delta": "#6baed6",
    "theta": "#74c476",
    "alpha": "#fd8d3c",
    "beta": "#9e9ac8",
    "gamma": "#f768a1",
}

# ================================================================
# 1.  PARÁMETROS FÍSICOS Y NUMÉRICOS
# ================================================================
N = 500
OMEGA0 = 0.0
GAMMA = 1.0
DT = 0.05
T_TRANS = 50.0
T_SIM = 100.0
ST = int(T_TRANS / DT)
SS = int(T_SIM / DT)
K_VALS = np.linspace(0.0, 6.0, 40)
KC_ANALITICO = 2.0 * GAMMA  # lorentziana exacto
T_ARR = np.arange(SS) * DT

# Valores de K para figuras cualitativas
K_BAJO = 0.8
K_CRIT = 2.0
K_ALTO = 4.5


# ================================================================
# 2.  DISTRIBUCIONES DE FRECUENCIAS
# ================================================================

def sample_lorentz(N, rng=RNG):
    u = rng.uniform(0, 1, N)
    return OMEGA0 + GAMMA * np.tan(np.pi * (u - 0.5))


def sample_gauss(N, rng=RNG):
    return rng.normal(OMEGA0, GAMMA, N)


def sample_uniform(N, rng=RNG):
    half = np.sqrt(3) * GAMMA
    return rng.uniform(OMEGA0 - half, OMEGA0 + half, N)


DISTS = {
    "Lorentziana": (sample_lorentz, COL["lor"]),
    "Gaussiana": (sample_gauss, COL["gau"]),
    "Uniforme": (sample_uniform, COL["uni"]),
}


# ================================================================
# 3.  NÚCLEO: DERIVADA O(N) + RK4
# ================================================================

def derivada(theta, omega, K):
    """Calcula F(θ) usando campo medio — O(N)."""
    z = np.exp(1j * theta).mean()
    r = float(np.abs(z))
    psi = float(np.angle(z))
    return omega + K * r * np.sin(psi - theta), r, psi


def rk4(theta, omega, K, dt=DT):
    """Un paso RK4."""
    k1, r, _ = derivada(theta, omega, K)
    k2, _, _ = derivada(theta + .5 * dt * k1, omega, K)
    k3, _, _ = derivada(theta + .5 * dt * k2, omega, K)
    k4, _, _ = derivada(theta + dt * k3, omega, K)
    return theta + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), r


def simular(omega, K, guardar_theta=False):
    """
    Integra N osciladores con RK4.
    Retorna: r_mean, r_series, theta_fin, omega_eff
             (+ theta_hist si guardar_theta=True)
    """
    rng_ic = np.random.default_rng(SEED)
    theta = rng_ic.uniform(0, 2 * np.pi, len(omega))

    for _ in range(ST):  # transitorio
        theta, _ = rk4(theta, omega, K)

    r_series = np.empty(SS)
    theta_prev = theta.copy()
    dtheta_acc = np.zeros(len(omega))
    theta_hist = [] if guardar_theta else None

    for n in range(SS):
        theta, r = rk4(theta, omega, K)
        r_series[n] = r
        dphi = theta - theta_prev
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dtheta_acc += dphi
        theta_prev = theta.copy()
        if guardar_theta:
            theta_hist.append(theta.copy())

    omega_eff = dtheta_acc / (SS * DT)
    result = (r_series.mean(), r_series, theta, omega_eff)
    return result if not guardar_theta else result + (np.array(theta_hist),)


def r_anal(K, gamma=GAMMA):
    """Solución analítica lorentziana."""
    Kc = 2.0 * gamma
    return np.where(K > Kc, np.sqrt(np.maximum(0, 1.0 - Kc / K)), 0.0)


# ================================================================
# 4.  BARRIDO EN K (datos base para todas las figuras)
# ================================================================
print("=" * 60)
print("BARRIDO <r> vs K — 3 distribuciones (N=500)")
print("=" * 60)

resultados = {}
for nombre, (sampler, _) in DISTS.items():
    print(f"  Simulando {nombre}…")
    omega = sampler(N)
    r_vals = np.empty(len(K_VALS))
    for idx, K in enumerate(K_VALS):
        r_vals[idx], _, _, _ = simular(omega, K)
    resultados[nombre] = (omega, r_vals)
    print(f"    Kc numérico estimado ≈ "
          f"{K_VALS[np.argmax(np.gradient(r_vals, K_VALS))]:.2f}")

# ================================================================
# 5.  FIGURA 1 — <r> vs K + predicción analítica
# ================================================================
print("\nFIG 1: <r> vs K …")
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
K_fine = np.linspace(0, 6, 500)

for nombre, (_, color) in DISTS.items():
    _, r_vals = resultados[nombre]
    ax1.plot(K_VALS, r_vals, "o-", color=color, ms=4,
             label=nombre, markevery=2)

ax1.plot(K_fine, r_anal(K_fine), "--",
         color=COL["anal"], lw=2.2,
         label=r"Analítica (Lor.): $r=\sqrt{1-K_c/K}$")
ax1.axvline(KC_ANALITICO, color=COL["Kc"], ls=":", lw=1.6,
            label=f"$K_c^{{\\rm Lor}}=2\\gamma={KC_ANALITICO:.1f}$")
ax1.set_xlabel(r"Acoplamiento $K$")
ax1.set_ylabel(r"$\langle r \rangle$")
ax1.set_title(r"Curva de sincronización $\langle r\rangle$ vs $K$")
ax1.set_xlim(0, 6);
ax1.set_ylim(-0.02, 1.05)
ax1.legend(framealpha=0.9)
fig1.tight_layout()
fig1.savefig("fig1_r_vs_K.png", dpi=150, bbox_inches="tight")
print("  → fig1_r_vs_K.png")

# ================================================================
# 6.  FIGURA 2 — r(t) para K bajo / crítico / alto
# ================================================================
print("FIG 2: r(t) temporal …")
omega_lor = resultados["Lorentziana"][0]
etiq_K = [
    rf"$K={K_BAJO}$ — incoherente ($K<K_c$)",
    rf"$K={K_CRIT}$ — crítico ($K\approx K_c$)",
    rf"$K={K_ALTO}$ — sincronizado ($K>K_c$)",
]
cols_K = ["#4393c3", "#f4a582", "#d6604d"]

fig2, axes2 = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for ax, K, lbl, col in zip(axes2, [K_BAJO, K_CRIT, K_ALTO], etiq_K, cols_K):
    _, r_ser, _, _ = simular(omega_lor, K)
    ax.plot(T_ARR, r_ser, color=col, lw=1.1, alpha=0.85)
    ax.axhline(r_ser.mean(), color="k", ls="--", lw=1.3,
               label=rf"$\langle r\rangle={r_ser.mean():.3f}$")
    ax.set_ylabel(r"$r(t)$");
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(lbl);
    ax.legend(loc="upper right")
axes2[-1].set_xlabel(r"Tiempo $t$")
fig2.suptitle("Evolución temporal — Lorentziana", y=1.01)
fig2.tight_layout()
fig2.savefig("fig2_r_temporal.png", dpi=150, bbox_inches="tight")
print("  → fig2_r_temporal.png")

# ================================================================
# 7.  FIGURA 3 — Snapshots circulares S^1
# ================================================================
print("FIG 3: Diagramas circulares …")
fig3, axes3 = plt.subplots(2, 3, figsize=(11, 6.5),
                           subplot_kw={"projection": "polar"})
for ci, (nombre, (_, color)) in enumerate(DISTS.items()):
    omega = resultados[nombre][0]
    for ri, K in enumerate([K_BAJO, K_ALTO]):
        _, _, theta_fin, _ = simular(omega, K)
        ax = axes3[ri, ci]
        ax.scatter(theta_fin, np.ones(N), s=6,
                   color=color, alpha=0.45, linewidths=0)
        z = np.exp(1j * theta_fin).mean()
        r_snap = abs(z);
        psi_snap = np.angle(z)
        ax.plot([psi_snap] * 2, [0, r_snap], color="k", lw=2.5)
        ax.plot(psi_snap, r_snap, "k^", ms=8)
        lbl_K = "incoherente" if ri == 0 else "sincronizado"
        ax.set_title(f"{nombre}\n$K={K}$  ({lbl_K})\n$r={r_snap:.3f}$",
                     fontsize=8, pad=8)
        ax.set_rticks([]);
        ax.tick_params(labelsize=7)
fig3.suptitle("Distribución de fases en $\\mathbb{S}^1$",
              fontsize=13, y=1.01)
fig3.tight_layout()
fig3.savefig("fig3_circular_snapshots.png", dpi=150, bbox_inches="tight")
print("  → fig3_circular_snapshots.png")

# ================================================================
# 8.  FIGURA 4 — Frecuencias efectivas Ω_i vs ω_i
# ================================================================
print("FIG 4: Frecuencias efectivas …")
fig4, axes4 = plt.subplots(2, 3, figsize=(11, 6))
for ci, (nombre, (_, color)) in enumerate(DISTS.items()):
    omega = resultados[nombre][0]
    for ri, K in enumerate([K_BAJO, K_ALTO]):
        _, _, _, oeff = simular(omega, K)
        ax = axes4[ri, ci]
        idx_s = np.argsort(omega)
        om_s = omega[idx_s];
        oef_s = oeff[idx_s]
        ax.scatter(om_s, oef_s, s=5, color=color, alpha=0.6)
        lim = [om_s.min(), om_s.max()]
        ax.plot(lim, lim, "k--", lw=1, label=r"$\Omega_i=\omega_i$")
        ax.set_title(f"{nombre}  $K={K}$\n"
                     f"({'incoherente' if ri == 0 else 'sincronizado'})",
                     fontsize=9)
        ax.set_xlabel(r"$\omega_i$", fontsize=9)
        ax.set_ylabel(r"$\langle\dot\theta_i\rangle_t$", fontsize=9)
        ax.legend(fontsize=7)
fig4.suptitle(r"Frecuencias efectivas $\Omega_i$ vs frecuencias naturales $\omega_i$",
              fontsize=12)
fig4.tight_layout()
fig4.savefig("fig4_freq_efectivas.png", dpi=150, bbox_inches="tight")
print("  → fig4_freq_efectivas.png")

# ================================================================
# 9.  FIGURA 5 — Espectro FFT de r(t)
# ================================================================
print("FIG 5: Espectro FFT …")
fig5, axes5 = plt.subplots(1, 3, figsize=(11, 4))
etiq_reg = [r"$K<K_c$", r"$K\approx K_c$", r"$K>K_c$"]
for ax, K, col, etq in zip(axes5, [K_BAJO, K_CRIT, K_ALTO],
                           cols_K, etiq_reg):
    _, r_ser, _, _ = simular(omega_lor, K)
    fluct = r_ser - r_ser.mean()
    freqs = rfftfreq(SS, d=DT)
    poder = (2. / SS) * np.abs(rfft(fluct))
    ax.plot(freqs, poder, color=col, lw=1.2)
    ax.set_xlabel(r"$f$ [u.a.]")
    ax.set_ylabel(r"$|\hat{R}(f)|$")
    ax.set_title(f"$K={K}$  ({etq})", fontsize=10)
    ax.set_xlim(0, 1.5)
fig5.suptitle("Espectro de potencia FFT de $r(t)-\\langle r\\rangle$ — Lorentziana",
              fontsize=12)
fig5.tight_layout()
fig5.savefig("fig5_fft_espectro.png", dpi=150, bbox_inches="tight")
print("  → fig5_fft_espectro.png")

# ================================================================
# 10.  FIGURA 6 — Validación numérica vs analítica
# ================================================================
print("FIG 6: Validación analítica …")
_, r_lor = resultados["Lorentziana"]
r_an = r_anal(K_VALS)
residuos = r_lor - r_an
mask = K_VALS > KC_ANALITICO
rms = np.sqrt(np.mean(residuos[mask] ** 2))

fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(7, 6),
                                  gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
ax6a.plot(K_VALS, r_lor, "o", color=COL["lor"], ms=5,
          label=f"Numérico RK4 ($N={N}$)")
ax6a.plot(K_fine, r_anal(K_fine), "-", color=COL["anal"], lw=2,
          label=r"Analítico: $r=\sqrt{1-K_c/K}$")
ax6a.axvline(KC_ANALITICO, color=COL["Kc"], ls=":", lw=1.5,
             label=f"$K_c={KC_ANALITICO:.1f}$")
ax6a.set_ylabel(r"$\langle r\rangle$")
ax6a.set_title("Validación numérica vs analítica — Distribución Lorentziana")
ax6a.legend();
ax6a.set_ylim(-0.02, 1.05)

dK = K_VALS[1] - K_VALS[0]
ax6b.bar(K_VALS, residuos, width=dK * 0.8,
         color=COL["lor"], alpha=0.7)
ax6b.axhline(0, color="k", lw=0.8)
ax6b.set_xlabel(r"$K$")
ax6b.set_ylabel(r"$r_{\rm num}-r_{\rm anal}$")
ax6b.set_title("Residuos", fontsize=10)
ax6b.text(0.98, 0.85, f"RMS$={rms:.4f}$",
          transform=ax6b.transAxes, ha="right",
          fontsize=9, color="darkred")
fig6.tight_layout()
fig6.savefig("fig6_validacion.png", dpi=150, bbox_inches="tight")
print(f"  → fig6_validacion.png  (RMS={rms:.5f})")

# ================================================================
# 11.  FIGURA 7 — Analogía con bandas EEG / ritmos neuronales
# ================================================================
print("FIG 7: Analogía bandas EEG …")


# Mapeo cualitativo: K → régimen neuronal
# Usamos la lorentziana como modelo
# Generamos señales EEG sintéticas sumando osciladores
def senal_eeg_sintetica(theta_hist, escala=1.0):
    """Suma de N señales de amplitud 1/N — campo eléctrico colectivo."""
    return escala * np.cos(theta_hist).mean(axis=1)


K_neural = {
    "Reposo\n(incoherente)\n$\\alpha$: 8–13 Hz": (K_BAJO, COL["alpha"]),
    "Transición\ncognitiva\n$\\beta$: 13–30 Hz": (K_CRIT, COL["beta"]),
    "Hipersincronía\n(epilepsia)\n$\\gamma>$30 Hz": (K_ALTO, COL["gamma"]),
}

fig7 = plt.figure(figsize=(13, 8))
gs7 = gridspec.GridSpec(3, 3, figure=fig7,
                        hspace=0.55, wspace=0.35)

t_eeg = T_ARR  # eje temporal compartido

for col_idx, (lbl, (K, col)) in enumerate(K_neural.items()):
    print(f"    Simulando K={K} (neuronal)…")
    _, r_ser, _, _, th_hist = simular(omega_lor, K, guardar_theta=True)
    eeg = senal_eeg_sintetica(th_hist)

    # Espectro Welch (más suave que FFT directa)
    f_w, psd = welch(eeg, fs=1. / DT, nperseg=256)

    # ── Fila 0: señal EEG sintética ──
    ax_eeg = fig7.add_subplot(gs7[0, col_idx])
    t_show = T_ARR[:600]
    ax_eeg.plot(t_show, eeg[:600], color=col, lw=0.8, alpha=0.9)
    ax_eeg.set_title(lbl, fontsize=8.5)
    ax_eeg.set_xlabel("$t$", fontsize=8)
    ax_eeg.set_ylabel("$V_{\\rm col}$", fontsize=8)
    ax_eeg.tick_params(labelsize=7)

    # ── Fila 1: r(t) ──
    ax_r = fig7.add_subplot(gs7[1, col_idx])
    ax_r.plot(T_ARR, r_ser, color=col, lw=1.1, alpha=0.85)
    ax_r.axhline(r_ser.mean(), color="k", ls="--", lw=1.1,
                 label=f"$\\langle r\\rangle={r_ser.mean():.3f}$")
    ax_r.set_ylim(-0.02, 1.05)
    ax_r.set_xlabel("$t$", fontsize=8)
    ax_r.set_ylabel("$r(t)$", fontsize=8)
    ax_r.legend(fontsize=7, loc="upper right")
    ax_r.tick_params(labelsize=7)

    # ── Fila 2: PSD ──
    ax_psd = fig7.add_subplot(gs7[2, col_idx])
    mask_f = f_w < 1.5
    ax_psd.semilogy(f_w[mask_f], psd[mask_f], color=col, lw=1.3)
    ax_psd.set_xlabel("$f$ [u.a.]", fontsize=8)
    ax_psd.set_ylabel("PSD", fontsize=8)
    ax_psd.set_title("Espectro de potencia", fontsize=8)
    ax_psd.tick_params(labelsize=7)

fig7.suptitle(
    "Analogía con ritmos neuronales: señal colectiva, parámetro de orden y PSD\n"
    "(Lorentziana, $N=500$, $\\gamma=1$)",
    fontsize=11, fontweight="bold")
fig7.savefig("fig7_neural_bandas.png", dpi=150, bbox_inches="tight")
print("  → fig7_neural_bandas.png")

# ================================================================
# 12.  FIGURA 8 — Panel neuronal completo (figura artículo)
# ================================================================
print("FIG 8: Panel neural completo …")

# Tabla de bandas EEG con interpretación
BANDAS_EEG = {
    "delta (0.5-4 Hz)": ("Sueno profundo", "Hipersincronizacion r->1", COL["delta"]),
    "theta (4-8 Hz)": ("Memoria/sueno REM", "Sincronizacion moderada", COL["theta"]),
    "alpha (8-13 Hz)": ("Reposo vigil", "Incoherencia parcial", COL["alpha"]),
    "beta (13-30 Hz)": ("Alerta/cognitivo", "Transicion K aprox Kc", COL["beta"]),
    "gamma (>30 Hz)": ("Proc. cognitivo", "Sincronizacion local", COL["gamma"]),
}

fig8, ax8 = plt.subplots(figsize=(10, 4.5))
ax8.axis("off")

col_labels = ["Banda EEG", "Estado cerebral",
              "Régimen Kuramoto", "Color"]
row_data = []
for banda, (estado, kuramoto, col) in BANDAS_EEG.items():
    row_data.append([banda, estado, kuramoto])

tabla = ax8.table(
    cellText=row_data,
    colLabels=["Banda EEG", "Estado cerebral", "Régimen Kuramoto"],
    cellLoc="center", loc="center",
    bbox=[0, 0, 1, 1]
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(11)

# Colorear filas según banda
for (row, col_idx), cell in tabla.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c2c2c")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        banda_col = list(BANDAS_EEG.values())[row - 1][2]
        cell.set_facecolor(banda_col + "40")  # 25% opacidad
        cell.set_edgecolor("#cccccc")

ax8.set_title("Tabla de analogías: Ritmos EEG ↔ Modelo de Kuramoto",
              fontsize=13, fontweight="bold", pad=20)
fig8.tight_layout()
fig8.savefig("fig8_panel_neural.png", dpi=150, bbox_inches="tight")
print("  → fig8_panel_neural.png")

# ================================================================
# 13.  FIGURA 9 — Panel resumen del artículo (multipanel)
# ================================================================
print("FIG 9: Panel resumen …")
fig9 = plt.figure(figsize=(14, 9))
gs9 = gridspec.GridSpec(2, 4, figure=fig9,
                        hspace=0.40, wspace=0.38)

# A: <r> vs K
axA = fig9.add_subplot(gs9[0, :2])
for nombre, (_, color) in DISTS.items():
    _, r_vals = resultados[nombre]
    axA.plot(K_VALS, r_vals, "o-", color=color, ms=3.5,
             label=nombre, markevery=2)
axA.plot(K_fine, r_anal(K_fine), "--", color=COL["anal"],
         lw=2, label="Analítica (Lor.)")
axA.axvline(KC_ANALITICO, color=COL["Kc"], ls=":", lw=1.4)
axA.text(KC_ANALITICO + 0.07, 0.06, r"$K_c$", fontsize=9, color="gray")
axA.set_xlabel(r"$K$");
axA.set_ylabel(r"$\langle r\rangle$")
axA.set_title(r"(A) $\langle r\rangle$ vs $K$")
axA.legend(fontsize=8, ncol=2)
axA.set_xlim(0, 6);
axA.set_ylim(-0.02, 1.05)

# B: g(ω) distribuciones
axB = fig9.add_subplot(gs9[0, 2])
w_plot = np.linspace(-5, 5, 400)
axB.plot(w_plot, GAMMA / np.pi / (w_plot ** 2 + GAMMA ** 2),
         color=COL["lor"], lw=1.8, label="Lorentziana")
axB.plot(w_plot, np.exp(-w_plot ** 2 / (2 * GAMMA ** 2)) / (GAMMA * np.sqrt(2 * np.pi)),
         color=COL["gau"], lw=1.8, ls="--", label="Gaussiana")
half = np.sqrt(3) * GAMMA
axB.plot([-half, -half, half, half],
         [0, 1 / (2 * half), 1 / (2 * half), 0],
         color=COL["uni"], lw=1.8, ls=":", label="Uniforme")
axB.set_xlabel(r"$\omega$");
axB.set_ylabel(r"$g(\omega)$")
axB.set_title("(B) Distribuciones");
axB.legend(fontsize=7)

# C: Validación lorentziana
axC = fig9.add_subplot(gs9[0, 3])
axC.plot(K_VALS, r_lor, "o", color=COL["lor"], ms=4,
         label=f"Numérico $N={N}$")
axC.plot(K_fine, r_anal(K_fine), "-", color=COL["anal"],
         lw=1.8, label="Analítico")
axC.axvline(KC_ANALITICO, color=COL["Kc"], ls=":", lw=1.2)
axC.set_xlabel(r"$K$");
axC.set_ylabel(r"$\langle r\rangle$")
axC.set_title(f"(C) Validación\nRMS$={rms:.4f}$")
axC.legend(fontsize=7)

# D: r(t) tres K
axD = fig9.add_subplot(gs9[1, :2])
for K, col, lbl in zip([K_BAJO, K_CRIT, K_ALTO], cols_K,
                       [f"$K={K_BAJO}$", f"$K={K_CRIT}$",
                        f"$K={K_ALTO}$"]):
    _, r_s, _, _ = simular(omega_lor, K)
    axD.plot(T_ARR, r_s, color=col, lw=1.0, alpha=0.85, label=lbl)
axD.set_xlabel(r"$t$");
axD.set_ylabel(r"$r(t)$")
axD.set_title("(D) Evolución temporal — Lorentziana")
axD.legend(fontsize=8);
axD.set_ylim(-0.02, 1.05)

# E: Snapshot circular K alto
axE = fig9.add_subplot(gs9[1, 2], projection="polar")
_, _, th_snap, _ = simular(omega_lor, K_ALTO)
axE.scatter(th_snap, np.ones(N), s=5, color=COL["lor"], alpha=0.45)
z_s = np.exp(1j * th_snap).mean()
axE.plot([np.angle(z_s)] * 2, [0, abs(z_s)], "k-", lw=2.5)
axE.plot(np.angle(z_s), abs(z_s), "k^", ms=8)
axE.set_rticks([])
axE.set_title(f"(E) $K={K_ALTO}$, $r={abs(z_s):.2f}$",
              fontsize=9, pad=10)

# F: Frecuencias efectivas K alto
axF = fig9.add_subplot(gs9[1, 3])
_, _, _, oeff = simular(omega_lor, K_ALTO)
idx_s = np.argsort(omega_lor)
om_s = omega_lor[idx_s];
oef_s = oeff[idx_s]
axF.scatter(om_s, oef_s, s=4, color=COL["lor"], alpha=0.6)
axF.plot([om_s.min(), om_s.max()],
         [om_s.min(), om_s.max()], "k--", lw=1,
         label=r"$\Omega_i=\omega_i$")
axF.set_xlabel(r"$\omega_i$");
axF.set_ylabel(r"$\Omega_i$")
axF.set_title(f"(F) Frec. efectivas $K={K_ALTO}$")
axF.legend(fontsize=7)

fig9.suptitle("Panel de resultados — Modelo de Kuramoto ($N=500$)",
              fontsize=14, fontweight="bold")
fig9.savefig("fig9_resumen.png", dpi=150, bbox_inches="tight")
print("  → fig9_resumen.png")

# ================================================================
# 14.  ANIMACIÓN 1 — Fases en S^1 evolucionando mientras K crece
# ================================================================
print("\nANIM 1: Fases en S^1 vs K …")

K_ANIM = np.linspace(0.2, 5.5, 50)
# Pre-calcular snapshots de theta para cada K
theta_snaps = []
r_snaps = []
for K in K_ANIM:
    _, _, th, _ = simular(omega_lor, K)
    theta_snaps.append(th.copy())
    r_snaps.append(abs(np.exp(1j * th).mean()))

fig_a1, ax_a1 = plt.subplots(figsize=(5.5, 5.5),
                             subplot_kw={"projection": "polar"})
fig_a1.patch.set_facecolor("#0d0d1a")
ax_a1.set_facecolor("#0d0d1a")
ax_a1.tick_params(colors="white", labelsize=8)
ax_a1.spines["polar"].set_color("#444466")

scat_a1, = ax_a1.plot([], [], "o", ms=4, color="#6eb4f7",
                      alpha=0.6, markeredgewidth=0)
arrow_a1, = ax_a1.plot([], [], "-", color="#ff6b6b", lw=3)
dot_a1, = ax_a1.plot([], [], "o", color="#ff6b6b", ms=10)
ax_a1.set_rticks([])
ax_a1.set_ylim(0, 1.05)
title_a1 = ax_a1.set_title("", color="white", fontsize=12, pad=15)
r_text_a1 = ax_a1.text(0.5, -0.10, "", transform=ax_a1.transAxes,
                       ha="center", fontsize=11, color="#ffd700")


def init_a1():
    scat_a1.set_data([], [])
    arrow_a1.set_data([], [])
    dot_a1.set_data([], [])
    return scat_a1, arrow_a1, dot_a1


def update_a1(frame):
    th = theta_snaps[frame]
    r = r_snaps[frame]
    K = K_ANIM[frame]
    z = np.exp(1j * th).mean()
    psi = np.angle(z)
    scat_a1.set_data(th, np.ones(N))
    arrow_a1.set_data([psi, psi], [0, r])
    dot_a1.set_data([psi], [r])
    title_a1.set_text(f"Fases en $\\mathbb{{S}}^1$ — $K={K:.2f}$")
    r_text_a1.set_text(f"$r = {r:.3f}$   |   "
                       + ("incoherente" if K < KC_ANALITICO
                          else "sincronizado"))
    return scat_a1, arrow_a1, dot_a1


anim1 = animation.FuncAnimation(
    fig_a1, update_a1, init_func=init_a1,
    frames=len(K_ANIM), interval=120, blit=True)
anim1.save("anim1_sincronizacion.gif",
           writer="pillow", fps=8, dpi=110)
plt.close(fig_a1)
print("  → anim1_sincronizacion.gif")

# ================================================================
# 15.  ANIMACIÓN 2 — r(t) en tiempo real para K creciente
# ================================================================
print("ANIM 2: r(t) animado …")

K_STEPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
series_anim2 = []
for K in K_STEPS:
    _, r_s, _, _ = simular(omega_lor, K)
    series_anim2.append(r_s[:600])  # primeros 600 pasos

WINDOW = 200  # ventana móvil

fig_a2, (ax_a2_r, ax_a2_K) = plt.subplots(
    2, 1, figsize=(8, 5),
    gridspec_kw={"height_ratios": [3, 1]})
fig_a2.patch.set_facecolor("#f8f8f8")

line_r, = ax_a2_r.plot([], [], color="#1f77b4", lw=1.6)
hline_r, = ax_a2_r.plot([], [], "r--", lw=1.2)
ax_a2_r.set_xlim(0, WINDOW * DT)
ax_a2_r.set_ylim(-0.02, 1.05)
ax_a2_r.set_ylabel(r"$r(t)$", fontsize=12)
ax_a2_r.axhline(0, color="gray", lw=0.5)
title_a2 = ax_a2_r.set_title("", fontsize=12)
rmean_txt = ax_a2_r.text(0.02, 0.92, "",
                         transform=ax_a2_r.transAxes, fontsize=10)

bar_K = ax_a2_K.barh([0], [K_STEPS[0]], height=0.5,
                     color="#ff7f0e", alpha=0.8)
ax_a2_K.axvline(KC_ANALITICO, color="red", ls="--", lw=1.3)
ax_a2_K.text(KC_ANALITICO + 0.05, 0.3, r"$K_c$", fontsize=9, color="red")
ax_a2_K.set_xlim(0, K_STEPS[-1] + 0.5)
ax_a2_K.set_yticks([])
ax_a2_K.set_xlabel(r"Acoplamiento $K$", fontsize=11)

total_frames = len(K_STEPS) * WINDOW


def update_a2(frame):
    k_idx = frame // WINDOW
    t_idx = frame % WINDOW
    if k_idx >= len(K_STEPS):
        return line_r, hline_r, rmean_txt
    r_s = series_anim2[k_idx]
    K = K_STEPS[k_idx]
    t_show = T_ARR[:t_idx + 1]
    r_show = r_s[:t_idx + 1]
    line_r.set_data(t_show, r_show)
    mean_v = r_show.mean()
    hline_r.set_data([0, t_show[-1]], [mean_v, mean_v])
    title_a2.set_text(
        f"Evolución de $r(t)$ — $K={K:.1f}$  "
        + ("(incoherente)" if K < KC_ANALITICO else "(sincronizado)"))
    rmean_txt.set_text(f"$\\langle r\\rangle = {mean_v:.3f}$")
    bar_K[0].set_width(K)
    return line_r, hline_r, rmean_txt


anim2 = animation.FuncAnimation(
    fig_a2, update_a2,
    frames=total_frames, interval=25, blit=False)
anim2.save("anim2_transicion_rt.gif",
           writer="pillow", fps=25, dpi=110)
plt.close(fig_a2)
print("  → anim2_transicion_rt.gif")

# ================================================================
# 16.  ANIMACIÓN 3 — Señal EEG colectiva + r(t) (neuronal)
# ================================================================
print("ANIM 3: Señal EEG neuronal …")

K_EEG_VALS = [K_BAJO, K_CRIT, K_ALTO]
LABEL_ESTADO = ["Reposo cortical\n(incoherente)",
                "Transición cognitiva\n(crítico)",
                "Crisis epiléptica\n(hipersincronía)"]
COL_EEG = [COL["alpha"], COL["beta"], COL["gamma"]]

# Pre-calcular trayectorias completas
th_hists, r_sers_eeg, eegs_all = [], [], []
for K in K_EEG_VALS:
    print(f"    Guardando trayectoria K={K}…")
    _, r_s, _, _, th_h = simular(omega_lor, K, guardar_theta=True)
    eeg = senal_eeg_sintetica(th_h, escala=1.0)
    th_hists.append(th_h)
    r_sers_eeg.append(r_s)
    eegs_all.append(eeg)

WIN_EEG = 300

fig_a3 = plt.figure(figsize=(12, 7))
gs_a3 = gridspec.GridSpec(3, 3, figure=fig_a3,
                          hspace=0.50, wspace=0.35)
fig_a3.patch.set_facecolor("#0a0a14")

ax_circ = [fig_a3.add_subplot(gs_a3[i, 0],
                              projection="polar") for i in range(3)]
ax_eeg_a = [fig_a3.add_subplot(gs_a3[i, 1]) for i in range(3)]
ax_r_a = [fig_a3.add_subplot(gs_a3[i, 2]) for i in range(3)]

lines_eeg, lines_r, scats, arrows, dots = [], [], [], [], []

for i, (K, col, lbl) in enumerate(zip(K_EEG_VALS, COL_EEG,
                                      LABEL_ESTADO)):
    # Circular
    ax_circ[i].set_facecolor("#0a0a14")
    ax_circ[i].spines["polar"].set_color("#333355")
    ax_circ[i].set_rticks([])
    ax_circ[i].tick_params(colors="#888899", labelsize=6)
    sc, = ax_circ[i].plot([], [], "o", ms=3,
                          color=col, alpha=0.5, markeredgewidth=0)
    arr, = ax_circ[i].plot([], [], "-", color="white", lw=2.5)
    dt, = ax_circ[i].plot([], [], "o", color="white", ms=8)
    ax_circ[i].set_title(lbl, color="white", fontsize=7, pad=6)
    scats.append(sc);
    arrows.append(arr);
    dots.append(dt)

    # EEG
    ax_eeg_a[i].set_facecolor("#0a0a14")
    ax_eeg_a[i].tick_params(colors="#aaaacc", labelsize=7)
    for sp in ax_eeg_a[i].spines.values():
        sp.set_color("#333355")
    ax_eeg_a[i].set_xlim(0, WIN_EEG * DT)
    ax_eeg_a[i].set_ylim(-1.1, 1.1)
    ax_eeg_a[i].set_ylabel("$V_{\\rm col}$", color="#aaaacc",
                           fontsize=8)
    if i == 0:
        ax_eeg_a[i].set_title("Señal EEG colectiva",
                              color="white", fontsize=9)
    le, = ax_eeg_a[i].plot([], [], color=col, lw=1.0)
    lines_eeg.append(le)

    # r(t)
    ax_r_a[i].set_facecolor("#0a0a14")
    ax_r_a[i].tick_params(colors="#aaaacc", labelsize=7)
    for sp in ax_r_a[i].spines.values():
        sp.set_color("#333355")
    ax_r_a[i].set_xlim(0, WIN_EEG * DT)
    ax_r_a[i].set_ylim(-0.02, 1.05)
    ax_r_a[i].set_ylabel("$r(t)$", color="#aaaacc", fontsize=8)
    ax_r_a[i].axhline(0.5, color="#444466", ls="--", lw=0.8)
    if i == 0:
        ax_r_a[i].set_title("Parámetro de orden",
                            color="white", fontsize=9)
    lr, = ax_r_a[i].plot([], [], color=col, lw=1.2)
    lines_r.append(lr)

title_a3 = fig_a3.suptitle(
    "Dinámica neuronal colectiva — Modelo de Kuramoto",
    color="white", fontsize=12, fontweight="bold")


def update_a3(frame):
    t_idx = min(frame, WIN_EEG - 1)
    t_show = T_ARR[:t_idx + 1]
    for i in range(3):
        th = th_hists[i][t_idx]
        z = np.exp(1j * th).mean()
        r = abs(z);
        psi = np.angle(z)
        scats[i].set_data(th, np.ones(N))
        arrows[i].set_data([psi, psi], [0, r])
        dots[i].set_data([psi], [r])
        lines_eeg[i].set_data(t_show, eegs_all[i][:t_idx + 1])
        lines_r[i].set_data(t_show, r_sers_eeg[i][:t_idx + 1])
    return (*scats, *arrows, *dots, *lines_eeg, *lines_r)


anim3 = animation.FuncAnimation(
    fig_a3, update_a3,
    frames=WIN_EEG, interval=40, blit=True)
anim3.save("anim3_neural_eeg.gif",
           writer="pillow", fps=20, dpi=100)
plt.close(fig_a3)
print("  → anim3_neural_eeg.gif")

# ================================================================
# 17.  RESUMEN FINAL
# ================================================================
print("\n" + "=" * 60)
print("SIMULACIÓN COMPLETADA")
print("=" * 60)
print(f"  N                 = {N}")
print(f"  γ (semiancho)     = {GAMMA}")
print(f"  Kc (Lorentziana)  = {KC_ANALITICO:.2f}")
print(f"  dt (RK4)          = {DT}")
print(f"  T_transitorio     = {T_TRANS}")
print(f"  T_simulación      = {T_SIM}")
print(f"  RMS validación    = {rms:.5f}")
print()
print("FIGURAS ESTÁTICAS:")
figs_desc = [
    ("fig1_r_vs_K.png", "<r> vs K — 3 distribuciones + analítica"),
    ("fig2_r_temporal.png", "r(t) para K bajo/crítico/alto"),
    ("fig3_circular_snapshots.png", "Snapshots de fases en S^1"),
    ("fig4_freq_efectivas.png", "Frecuencias efectivas Ω_i vs ω_i"),
    ("fig5_fft_espectro.png", "Espectro FFT de r(t)"),
    ("fig6_validacion.png", "Validación numérica vs analítica"),
    ("fig7_neural_bandas.png", "Señal EEG, r(t) y PSD — 3 regímenes"),
    ("fig8_panel_neural.png", "Tabla analogías EEG ↔ Kuramoto"),
    ("fig9_resumen.png", "Panel multipanel del artículo"),
]
for f, d in figs_desc:
    print(f"  {f:<38} — {d}")
print()
print("ANIMACIONES:")
anims_desc = [
    ("anim1_sincronizacion.gif", "Fases en S^1 mientras K crece"),
    ("anim2_transicion_rt.gif", "r(t) animado con K creciente"),
    ("anim3_neural_eeg.gif", "EEG colectivo + r(t) — 3 regímenes"),
]
for f, d in anims_desc:
    print(f"  {f:<38} — {d}")
print("=" * 60)