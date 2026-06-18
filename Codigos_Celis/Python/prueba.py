"""
═══════════════════════════════════════════════════════════════════════
  Evolución Temporal de un Paquete de Ondas Gaussiano con V(x) ≠ 0
  Metodo: Split-Step en espacio de Fourier — FFT O(N log N)
  Autores: Juan Carlos Celis Lozada | Andrés Eduardo Arrieta Lozano
  Curso  : Física Computacional II

  CORRECCIONES APLICADAS:
    1. np.trapz → np.trapezoid  (eliminado en NumPy 2.0)
    2. DFT/IDFT O(N²) manual → np.fft.fft / np.fft.ifft  O(N log N)
    3. phi_k guardado ANTES del 2do medio paso en x (espectro limpio)
    4. E_cin calculada del phi_k limpio (sin fase V absorbida)
    5. Comparación sigma(t) analítica marcada como válida solo para V=0
    6. Línea vertical t_llegada en gráfica T/R para contextualizar R≈1 inicial

  Potenciales implementados:
    1. Barrera gaussiana   → V(x) = V0 * exp(-x²/2d²)
    2. Pozo gaussiano      → V(x) = -V0 * exp(-x²/2d²)
    3. Escalón de potencial→ V(x) = V0/2 * (1 + tanh((x-x0)/a))
    4. Doble barrera       → V(x) = V0[exp(-(x-a)²/2d²) + exp(-(x+a)²/2d²)]

  El metodo split-step aplica:
    ψ(t+Δt) ≈ e^(-iVΔt/2ħ) · IFFT[e^(-iħk²Δt/2m) · FFT[e^(-iVΔt/2ħ)·ψ(t)]]
  Error de truncamiento: O(Δt²) por paso (Strang splitting de 2º orden).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import warnings
warnings.filterwarnings('ignore')

# ─── Compatibilidad NumPy 1.x / 2.x ─────────────────────────────────────────
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

# ─── Constantes (unidades naturales) ─────────────────────────────────────────
hbar = 1.0
m    = 1.0
pi   = np.pi

# ─── Parámetros de la grilla ─────────────────────────────────────────────────
N    = 256
L    = 30.0
dx   = 2*L / N
x    = np.linspace(-L, L, N, endpoint=False)

# ─── Parámetros del paquete inicial ──────────────────────────────────────────
sigma0 = 1.0
k0     = 3.0
x0_psi = -10.0

# ─── Parámetros temporales ───────────────────────────────────────────────────
dt      = 0.02
T_max   = 8.0
n_steps = int(T_max / dt)
t_arr   = np.arange(n_steps + 1) * dt

# ─── Grilla de momentos ──────────────────────────────────────────────────────
k_arr = np.fft.fftfreq(N, d=dx) * 2 * pi

# ─── Tiempo estimado de llegada a la barrera ─────────────────────────────────
# v_grupo = ħ·k0/m; el paquete parte en x0_psi < 0
t_llegada = abs(x0_psi) / (hbar * k0 / m)

# ══════════════════════════════════════════════════════════════════════════════
#   FFT / IFFT
# ══════════════════════════════════════════════════════════════════════════════

def dft(y):
    """Transforma ψ(x) → φ(k) usando fft compleja."""
    return np.fft.fft(y)

def idft(phi_k, N_loc=None):
    """Transforma φ(k) → ψ(x) usando ifft compleja."""
    return np.fft.ifft(phi_k)

# ══════════════════════════════════════════════════════════════════════════════
#   Propagadores
# ══════════════════════════════════════════════════════════════════════════════

def propagador_k(phi_k, k_arr, dt_prop):
    """Propaga en espacio-k: φ_k ← φ_k · exp(-i ħ k² Δt / 2m)."""
    return phi_k * np.exp(-1j * hbar * k_arr**2 / (2 * m) * dt_prop)

def propagador_x(psi, V, dt_prop, fraccion=1.0):
    """Propaga en espacio-x: ψ ← ψ · exp(-i V Δt frac / ħ)."""
    return psi * np.exp(-1j * V * dt_prop * fraccion / hbar)

# ══════════════════════════════════════════════════════════════════════════════
#   Definiciones de potenciales
# ══════════════════════════════════════════════════════════════════════════════

def potencial_barrera_gaussiana(x, V0=3.0, d=0.8, x_center=0.0):
    return V0 * np.exp(-(x - x_center)**2 / (2 * d**2))

def potencial_pozo_gaussiano(x, V0=3.0, d=1.5, x_center=0.0):
    return -V0 * np.exp(-(x - x_center)**2 / (2 * d**2))

def potencial_escalon(x, V0=2.0, x0=0.0, a=0.5):
    return (V0 / 2) * (1 + np.tanh((x - x0) / a))

def potencial_doble_barrera(x, V0=3.0, d=0.5, sep=3.0):
    return V0 * (np.exp(-(x - sep)**2 / (2 * d**2)) +
                 np.exp(-(x + sep)**2 / (2 * d**2)))

# ══════════════════════════════════════════════════════════════════════════════
#   Estado inicial
# ══════════════════════════════════════════════════════════════════════════════

def psi0_gauss(x, sigma, k0, x0=0.0):
    norm = (sigma**2 * pi)**(-0.25)
    return norm * np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

# ══════════════════════════════════════════════════════════════════════════════
#   Simulación split-step completa
# ══════════════════════════════════════════════════════════════════════════════

def simular_split_step(V, label=""):
    print(f"\n  Simulando: {label}")
    print(f"  N={N}, L={L}, σ₀={sigma0}, k₀={k0}, x₀={x0_psi}")
    print(f"  dt={dt}, T={T_max}, pasos={n_steps}")

    psi = psi0_gauss(x, sigma0, k0, x0=x0_psi)

    hist = {
        't'       : [],
        'psi'     : [],
        'prob'    : [],
        'xmean'   : [],
        'sigma_x' : [],
        'norma'   : [],
        'energia' : [],
        'phi_k'   : [],
        'J'       : [],
        'T_coef'  : [],
        'R_coef'  : [],
    }

    idx_sep = N // 2   # x[idx_sep] = 0.0 exacto con linspace(-L,L,N,endpoint=False)

    # ── FIX 1 y 2: registrar() acepta phi_k_pre (calculado ANTES del 2do medio
    #               paso en x) para obtener espectro y E_cin sin la fase de V.
    def registrar(t_val, psi_val, phi_k_pre=None):
        prob  = np.abs(psi_val)**2
        norma = np.trapezoid(prob, x)
        xm    = np.trapezoid(x * prob, x) / norma
        x2m   = np.trapezoid(x**2 * prob, x) / norma
        sig   = np.sqrt(max(x2m - xm**2, 0))

        # Corriente de probabilidad
        dpsi_dx = np.gradient(psi_val, dx)
        J_val   = (hbar / m) * np.imag(np.conj(psi_val) * dpsi_dx)

        # Coeficientes de transmisión y reflexión
        T_c = np.trapezoid(prob[idx_sep:], x[idx_sep:]) / norma
        R_c = np.trapezoid(prob[:idx_sep], x[:idx_sep]) / norma

        # ── FIX 1: usar phi_k limpio (sin fase V del 2do medio paso en x).
        #    Si no se pasa phi_k_pre (p.ej. en t=0), se recalcula de psi_val;
        #    en t=0 no hay medio paso previo así que el resultado es idéntico.
        phi_k_now = phi_k_pre if phi_k_pre is not None else dft(psi_val)

        # ── FIX 2: E_cin del phi_k limpio → sin error O(V·Δt) por paso.
        #    Por Parseval discreto: Σ|FFT(ψ)|² = N·Σ|ψ|² → N·norm_k = N·norma/dx
        #    → N·norm_k y norma_x comparten el mismo denominador efectivo.
        spec   = np.abs(phi_k_now)**2
        norm_k = np.sum(spec) / N
        if norm_k > 0:
            E_cin = (hbar**2 / (2*m)) * np.sum(k_arr**2 * spec) / (N * norm_k)
        else:
            E_cin = 0.0
        E_pot = np.trapezoid(V * prob, x) / norma
        E_tot = E_cin + E_pot

        hist['t'].append(t_val)
        hist['prob'].append(prob.copy())
        hist['psi'].append(psi_val.copy())
        hist['xmean'].append(xm)
        hist['sigma_x'].append(sig)
        hist['norma'].append(norma)
        hist['energia'].append(E_tot)
        hist['phi_k'].append(phi_k_now.copy())
        hist['J'].append(J_val.copy())
        hist['T_coef'].append(T_c)
        hist['R_coef'].append(R_c)

    # ── Bucle de evolución ───────────────────────────────────────────────────
    registrar(0.0, psi)   # t=0: phi_k_pre=None → dft(psi) sin fase extra

    for step in range(n_steps):
        # 1) Medio paso en x
        psi = propagador_x(psi, V, dt, fraccion=0.5)
        # 2) Paso completo en k
        phi_k = dft(psi)
        phi_k = propagador_k(phi_k, k_arr, dt)
        psi   = idft(phi_k)
        # ── FIX 1 y 2: capturar phi_k limpio ANTES del 2do medio paso en x
        phi_k_clean = dft(psi)
        # 3) Medio paso en x
        psi = propagador_x(psi, V, dt, fraccion=0.5)

        t_val = (step + 1) * dt
        if (step + 1) % max(1, n_steps // 80) == 0:
            registrar(t_val, psi, phi_k_pre=phi_k_clean)
            prog = (step + 1) / n_steps * 100
            if int(prog) % 20 == 0:
                print(f"    {prog:.0f}% — t={t_val:.2f}  "
                      f"norma={hist['norma'][-1]:.6f}  "
                      f"T={hist['T_coef'][-1]:.4f}  "
                      f"R={hist['R_coef'][-1]:.4f}")

    print(f"  ✓ Completado — T_final={hist['T_coef'][-1]:.4f}  "
          f"R_final={hist['R_coef'][-1]:.4f}")
    return hist

# ══════════════════════════════════════════════════════════════════════════════
#   Figuras
# ══════════════════════════════════════════════════════════════════════════════

COLOR_NUM = '#8B0000'
COLOR_POT = '#1a5276'
COLOR_AN  = '#555555'
COLOR_T   = '#28b463'
COLOR_R   = '#e67e22'
FSIZE     = (12, 8)


def fig_densidad_multipanel(hist, V, label, fname):
    """Figura 1: densidad de probabilidad en múltiples tiempos."""
    t_arr_h  = hist['t']
    prob_arr = hist['prob']
    n_total  = len(t_arr_h)
    indices  = [0,
                n_total // 5,
                2 * n_total // 5,
                3 * n_total // 5,
                4 * n_total // 5,
                n_total - 1]
    fig, axes = plt.subplots(2, 3, figsize=FSIZE)
    axes = axes.flatten()
    V_scaled = V / np.max(np.abs(V)) * 0.3 if np.max(np.abs(V)) > 0 else V

    for ax, idx in zip(axes, indices):
        t_val = t_arr_h[idx]
        prob  = prob_arr[idx]
        ax.fill_between(x, prob, alpha=0.35, color=COLOR_NUM)
        ax.plot(x, prob, color=COLOR_NUM, lw=1.5, label=r'$|\psi|^2$ FFT')
        ax.plot(x, V_scaled + prob.max() * 0.05,
                color=COLOR_POT, lw=1.2, ls='--',
                label='$V(x)$ (escala)', alpha=0.7)
        ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.set_title(f'$t = {t_val:.2f}$', fontsize=11)
        ax.set_xlabel('$x$', fontsize=10)
        ax.set_ylabel(r'$|\psi|^2$', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.set_xlim(-L, L)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Densidad de probabilidad — {label}\n'
                 r'$\sigma_0=' + f'{sigma0}$, $k_0={k0}$, $N={N}$, $L={L}$',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {fname}')


def fig_transmision_reflexion(hist, label, fname):
    """Figura 2: coeficientes T y R en función del tiempo."""
    t_h = hist['t']
    T_h = hist['T_coef']
    R_h = hist['R_coef']
    TR  = [t + r for t, r in zip(T_h, R_h)]

    fig, axes = plt.subplots(1, 2, figsize=FSIZE)

    ax = axes[0]
    ax.plot(t_h, T_h, color=COLOR_T, lw=2, label='Transmisión $T(t)$')
    ax.plot(t_h, R_h, color=COLOR_R, lw=2, label='Reflexión $R(t)$')
    ax.plot(t_h, TR,  color='purple', lw=1.5, ls='--',
            label='$T+R$', alpha=0.8)
    ax.axhline(1.0, color='gray', ls=':', lw=1)

    # ── FIX 4: marcar t_llegada para contextualizar R≈1 antes de la interacción
    ax.axvline(t_llegada, color='steelblue', ls='--', lw=1.3, alpha=0.8)
    ax.text(t_llegada + 0.08, 1.10,
            f'$t_{{arr}}\\approx{t_llegada:.1f}$',
            fontsize=8, color='steelblue', va='top')
    ax.annotate('llegada\na barrera',
                xy=(t_llegada, 0.5),
                xytext=(t_llegada - 1.5, 0.6),
                fontsize=7, color='steelblue',
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.9))

    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('Probabilidad', fontsize=12)
    ax.set_title('Coeficientes de transmisión y reflexión\n'
                 r'($R\approx 1$ antes de $t_{arr}$: correcto, paquete aún a izquierda)',
                 fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.20)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    norma_h = hist['norma']
    E_h     = hist['energia']
    ax2.plot(t_h, norma_h, color=COLOR_NUM, lw=2,
             label=r'Norma $\int|\psi|^2\,dx$')
    ax2_r = ax2.twinx()
    ax2_r.plot(t_h, E_h, color='darkorange', lw=2, ls='--',
               label=r'$\langle E\rangle$ total')
    ax2.set_xlabel('$t$', fontsize=12)
    ax2.set_ylabel('Norma', fontsize=12, color=COLOR_NUM)
    ax2_r.set_ylabel(r'$\langle E\rangle$', fontsize=12, color='darkorange')
    ax2.set_title('Conservación de norma y energía\n'
                  r'(deriva en $\langle E\rangle$: $\mathcal{O}(\Delta t^2)$ acumulado, esperado)',
                  fontsize=10)
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Transmisión / Reflexión — {label}', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {fname}')


def fig_observables_temporales(hist, label, fname):
    """Figura 3: ⟨x⟩, σ(t), norma y energía."""
    t_h   = hist['t']
    xm_h  = hist['xmean']
    sig_h = hist['sigma_x']
    n_h   = hist['norma']
    E_h   = hist['energia']

    fig, axes = plt.subplots(2, 2, figsize=FSIZE)

    axes[0, 0].plot(t_h, xm_h, color=COLOR_NUM, lw=2)
    axes[0, 0].set_xlabel('$t$')
    axes[0, 0].set_ylabel(r'$\langle x\rangle$')
    axes[0, 0].set_title('Posición media (Ehrenfest)')
    axes[0, 0].grid(True, alpha=0.3)

    # ── FIX 3: σ(t) analítico solo válido para V=0 → indicarlo explícitamente
    sig_an = [sigma0 * np.sqrt(1 + (hbar * ti / (m * sigma0**2))**2)
              for ti in t_h]
    axes[0, 1].plot(t_h, sig_h, color=COLOR_NUM, lw=2, label='FFT numérico')
    axes[0, 1].plot(t_h, sig_an, color=COLOR_AN, lw=1.5, ls='--',
                    alpha=0.45, label='Libre analítico (solo V=0)')
    # Anotación aclaratoria en la mitad del eje temporal
    mid = len(t_h) // 2
    axes[0, 1].annotate('referencia\nsolo para V=0',
                         xy=(t_h[mid], sig_an[mid]),
                         xytext=(t_h[mid] * 0.5, sig_an[mid] * 1.18),
                         fontsize=7, color=COLOR_AN, alpha=0.8,
                         arrowprops=dict(arrowstyle='->', color=COLOR_AN,
                                         lw=0.8, alpha=0.6))
    axes[0, 1].set_xlabel('$t$')
    axes[0, 1].set_ylabel(r'$\sigma(t)$')
    axes[0, 1].set_title('Ancho del paquete\n'
                          r'(divergencia numérica vs analítico: efecto físico real, no error)',
                          fontsize=9)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_h, n_h, color=COLOR_NUM, lw=2)
    axes[1, 0].axhline(1.0, color='gray', ls=':', lw=1)
    axes[1, 0].set_xlabel('$t$')
    axes[1, 0].set_ylabel('Norma')
    axes[1, 0].set_title('Conservación de norma')
    axes[1, 0].set_ylim(0.98, 1.02)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_h, E_h, color=COLOR_NUM, lw=2)
    axes[1, 1].set_xlabel('$t$')
    axes[1, 1].set_ylabel(r'$\langle E\rangle$')
    axes[1, 1].set_title(r'Energía total media''\n'
                          r'($E_{cin}$ calculada de $\phi_k$ limpio, sin fase $V$)',
                          fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'Observables temporales — {label}', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {fname}')


def fig_corriente(hist, V, label, fname):
    """Figura 4: corriente de probabilidad J(x,t)."""
    n_total = len(hist['t'])
    indices = [0, n_total//4, n_total//2, 3*n_total//4, n_total-1]
    colores = plt.cm.plasma(np.linspace(0.2, 0.9, len(indices)))

    fig, axes = plt.subplots(1, 2, figsize=FSIZE)
    V_n = V / np.max(np.abs(V)) if np.max(np.abs(V)) > 0 else V

    for idx, c in zip(indices, colores):
        t_v = hist['t'][idx]
        J_v = hist['J'][idx]
        axes[0].plot(x, J_v, color=c, lw=1.5, label=f'$t={t_v:.1f}$')

    axes[0].plot(x, 0.5 * V_n, color=COLOR_POT, lw=1.5, ls='--',
                 alpha=0.7, label='$V(x)$ norm.')
    axes[0].axvline(0, color='gray', ls=':', lw=0.8)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$J(x,t)$')
    axes[0].set_title('Corriente de probabilidad')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-L, L)

    J_matrix = np.array(hist['J'])
    t_plot   = hist['t']
    im = axes[1].pcolormesh(x, t_plot, J_matrix, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=axes[1], label='$J(x,t)$')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$t$')
    axes[1].set_title('Mapa de calor $J(x,t)$')

    fig.suptitle(f'Corriente de probabilidad — {label}', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {fname}')


def fig_comparacion_potenciales(hists, labels, Vs, fname):
    """Figura 5: comparación de T y R para todos los potenciales."""
    fig, axes = plt.subplots(2, 2, figsize=FSIZE)
    colores = ['#8B0000', '#1a5276', '#28b463', '#7d3c98']

    for i, (hist, lab, color) in enumerate(zip(hists, labels, colores)):
        ax = axes.flatten()[i]
        t_h = hist['t']
        ax.plot(t_h, hist['T_coef'], color=color, lw=2, label='Transmisión')
        ax.plot(t_h, hist['R_coef'], color=color, lw=2,
                ls='--', label='Reflexión', alpha=0.7)
        # Línea de llegada a la barrera
        ax.axvline(t_llegada, color='steelblue', ls=':', lw=1.0, alpha=0.7)
        ax.text(t_llegada + 0.05, 1.12,
                f'$t_{{arr}}$', fontsize=7, color='steelblue')
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel('$t$')
        ax.set_ylabel('Probabilidad')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.20)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Comparación T y R — Todos los potenciales', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {fname}')


def fig_potenciales(Vs, labels, fname):
    """Figura 6: todos los potenciales en una sola figura."""
    colores = ['#8B0000', '#1a5276', '#28b463', '#7d3c98']
    fig, axes = plt.subplots(2, 2, figsize=FSIZE)

    for ax, V, lab, c in zip(axes.flatten(), Vs, labels, colores):
        ax.plot(x, V, color=c, lw=2)
        ax.fill_between(x, V, alpha=0.2, color=c)
        Ek = (hbar**2 / (2*m)) * (k0**2 + 1/(2*sigma0**2))
        ax.axhline(Ek, color='gray', ls='--', lw=1.5,
                   label=f'$\\langle E\\rangle_{{cin}} = {Ek:.2f}$')
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$V(x)$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-L/2, L/2)

    fig.suptitle('Potenciales simulados', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {fname}')


def fig_densidad_final_comparada(hists, labels, Vs, fname):
    """Figura 7: densidad final |ψ|² para todos los potenciales."""
    colores = ['#8B0000', '#1a5276', '#28b463', '#7d3c98']
    fig, axes = plt.subplots(2, 2, figsize=FSIZE)

    for ax, hist, lab, V, c in zip(axes.flatten(), hists, labels, Vs, colores):
        prob_f = hist['prob'][-1]
        V_n    = V / np.max(np.abs(V)) * prob_f.max() * 0.8 \
                 if np.max(np.abs(V)) > 0 else V
        ax.fill_between(x, prob_f, alpha=0.35, color=c)
        ax.plot(x, prob_f, color=c, lw=2, label=r'$|\psi(x,T)|^2$')
        ax.plot(x, V_n, color=COLOR_POT, lw=1.5, ls='--',
                alpha=0.7, label='$V(x)$ (escala)')
        T_f = hist['T_coef'][-1]
        R_f = hist['R_coef'][-1]
        ax.set_title(f'{lab}\n$T={T_f:.3f}$,  $R={R_f:.3f}$', fontsize=9)
        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$|\psi|^2$')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlim(-L, L)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Densidad de probabilidad final $t={T_max}$', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {fname}')


def generar_gif(hist, V, label, fname):
    """Animación GIF de la evolución."""
    fig, ax = plt.subplots(figsize=(9, 5))
    V_n = V / np.max(np.abs(V)) * 0.4 if np.max(np.abs(V)) > 0 else V

    prob_max = max(np.max(p) for p in hist['prob']) * 1.15
    line_p,  = ax.plot([], [], color=COLOR_NUM, lw=2, label=r'$|\psi|^2$')
    ax.plot(x, V_n, color=COLOR_POT, lw=1.5, ls='--', alpha=0.7, label='$V(x)$')
    ax.set_xlim(-L, L)
    ax.set_ylim(0, prob_max)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$|\psi|^2$')
    ax.legend(loc='upper right', fontsize=9)
    title_txt = ax.set_title('')
    ax.grid(True, alpha=0.3)

    def init():
        line_p.set_data([], [])
        return line_p,

    def update(frame):
        prob = hist['prob'][frame]
        t_v  = hist['t'][frame]
        line_p.set_data(x, prob)
        T_v = hist['T_coef'][frame]
        R_v = hist['R_coef'][frame]
        title_txt.set_text(
            f'{label} — $t={t_v:.2f}$   '
            f'$T={T_v:.3f}$   $R={R_v:.3f}$')
        return line_p,

    ani = FuncAnimation(fig, update, frames=len(hist['t']),
                        init_func=init, blit=False, interval=80)
    writer = PillowWriter(fps=12)
    ani.save(fname, writer=writer)
    plt.close()
    print(f'  ✓ {fname}')


# ══════════════════════════════════════════════════════════════════════════════
#   EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Simulación con V(x) ≠ 0 — Split-Step FFT O(N log N)")
    print(f"  N={N}, L={L}, σ₀={sigma0}, k₀={k0}, x₀={x0_psi}")
    print(f"  dt={dt}, T={T_max}, pasos={n_steps}")
    print(f"  t_llegada estimado = {t_llegada:.2f}")
    print("=" * 65)

    # ── Definir los cuatro potenciales ───────────────────────────────────────
    V_barrera = potencial_barrera_gaussiana(x, V0=5.0, d=0.8)
    V_pozo    = potencial_pozo_gaussiano   (x, V0=4.0, d=1.5)
    V_escalon = potencial_escalon          (x, V0=3.0, x0=0.0, a=0.5)
    V_doble   = potencial_doble_barrera    (x, V0=5.0, d=0.5, sep=2.0)

    Vs     = [V_barrera, V_pozo, V_escalon, V_doble]
    labels = ['Barrera gaussiana ($V_0=5$)',
              'Pozo gaussiano ($V_0=4$)',
              'Escalón de potencial ($V_0=3$)',
              'Doble barrera ($V_0=5$)']
    tags   = ['barrera', 'pozo', 'escalon', 'doble_barrera']

    # ── Figura de los potenciales ─────────────────────────────────────────────
    fig_potenciales(Vs, labels, 'figV0_potenciales.png')

    # ── Simular y graficar cada caso ─────────────────────────────────────────
    hists = []
    for V, lab, tag in zip(Vs, labels, tags):
        hist = simular_split_step(V, label=lab)
        hists.append(hist)
        fig_densidad_multipanel     (hist, V, lab, f'figV_{tag}_densidad.png')
        fig_transmision_reflexion   (hist, lab,     f'figV_{tag}_TR.png')
        fig_observables_temporales  (hist, lab,     f'figV_{tag}_observables.png')
        fig_corriente               (hist, V, lab,  f'figV_{tag}_corriente.png')
        generar_gif                 (hist, V, lab,  f'figV_{tag}_animacion.gif')

    # ── Figuras comparativas ──────────────────────────────────────────────────
    fig_comparacion_potenciales(hists, labels, Vs, 'figV_comparacion_TR.png')
    fig_densidad_final_comparada(hists, labels, Vs, 'figV_densidad_final_comparada.png')

    # ── Tabla de resultados finales ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TABLA: Resultados finales — T, R, norma, ⟨E⟩")
    print("=" * 65)
    print(f"  {'Potencial':<35} {'T':>7} {'R':>7} {'T+R':>6} "
          f"{'Norma':>8} {'⟨E⟩':>8}")
    print("  " + "-" * 63)
    for hist, lab in zip(hists, labels):
        T_f = hist['T_coef'][-1]
        R_f = hist['R_coef'][-1]
        N_f = hist['norma'][-1]
        E_f = hist['energia'][-1]
        print(f"  {lab:<35} {T_f:>7.4f} {R_f:>7.4f} "
              f"{T_f+R_f:>6.4f} {N_f:>8.6f} {E_f:>8.4f}")
    print("=" * 65)
    print("\n  Archivos generados:")
    import glob
    for f in sorted(glob.glob('figV_*.png') + glob.glob('figV_*.gif')):
        print(f"    ✓  {f}")