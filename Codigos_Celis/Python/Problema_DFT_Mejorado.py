"""
=============================================================================
  Evolución Temporal de un Paquete de Ondas Gaussiano — VERSIÓN EXTENDIDA
  Autores: Juan Carlos Celis Lozada | Andrés Eduardo Arrieta Lozano
  Física Computacional II
  =============================================================================
  NUEVAS CARACTERÍSTICAS RESPECTO A LA VERSIÓN ORIGINAL:
    1. Potenciales V(x) ≠ 0: barrera rectangular, pozo armónico, doble ranura
    2. Metodo split-step completo de segundo orden (Strang splitting)
    3. Extensión a 2D con potenciales 2D
    4. Análisis de coeficientes de transmisión/reflexión
    5. Comparación entre distintos potenciales
  =============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import warnings
warnings.filterwarnings('ignore')

pi = np.pi

# ═══════════════════════════════════════════════════════════════════
#  PARÁMETROS GLOBALES (unidades naturales: ħ = m = 1)
# ═══════════════════════════════════════════════════════════════════
hbar = 1.0
m    = 1.0

# Grilla 1D
N    = 512          # puntos (más resolución para potenciales)
L    = 30.0         # dominio [-L, L]
x    = np.linspace(-L, L, N, endpoint=False)
dx   = x[1] - x[0]

# Grilla de momentos (fftfreq da frecuencias en ciclos/unidad → × 2π)
k_arr = 2*pi * np.fft.fftfreq(N, d=dx)

# Parámetros del paquete inicial
sigma0 = 1.0
k0     = 5.0        # momento inicial (más alto para ver efectos de barrera)
x0     = -12.0      # posición inicial (a la izquierda de la barrera)

# Tiempo
dt     = 0.01
T_max  = 8.0
n_steps = int(T_max / dt)
t_arr   = np.linspace(0, T_max, n_steps + 1)

# ═══════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════

def psi0_gauss(x, x0=0.0, sigma0=1.0, k0=3.0):
    """Estado inicial gaussiano normalizado centrado en x0."""
    N_norm = (pi * sigma0**2)**(-0.25)
    return N_norm * np.exp(-(x - x0)**2 / (2*sigma0**2)) * np.exp(1j*k0*x)


def normalize(psi, dx):
    """Normaliza la función de onda."""
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm


def energy_kinetic(psi, k_arr, dx):
    """Energía cinética via espacio k."""
    phi = np.fft.fft(psi)
    E_k = hbar**2 * k_arr**2 / (2*m)
    return np.real(np.sum(np.conj(phi) * E_k * phi)) * dx / N


def mean_position(psi, x, dx):
    return np.real(np.sum(x * np.abs(psi)**2) * dx)


def sigma_width(psi, x, dx):
    xm = mean_position(psi, x, dx)
    x2m = np.real(np.sum(x**2 * np.abs(psi)**2) * dx)
    return np.sqrt(max(x2m - xm**2, 0.0))


# ═══════════════════════════════════════════════════════════════════
#  DEFINICIÓN DE POTENCIALES 1D
# ═══════════════════════════════════════════════════════════════════

def V_free(x):
    """Partícula libre: V = 0."""
    return np.zeros_like(x)


def V_barrier(x, V0=30.0, x_c=0.0, width=1.0):
    """Barrera de potencial rectangular."""
    V = np.zeros_like(x)
    mask = np.abs(x - x_c) <= width/2
    V[mask] = V0
    return V


def V_harmonic(x, omega=1.0, x_c=0.0):
    """Pozo armónico centrado en x_c."""
    return 0.5 * m * omega**2 * (x - x_c)**2


def V_double_well(x, V0=20.0, a=3.0):
    """Doble pozo: W-shaped potential."""
    return V0 * ((x/a)**2 - 1)**2


def V_step(x, V0=15.0, x_c=0.0):
    """Escalón de potencial."""
    V = np.zeros_like(x)
    V[x > x_c] = V0
    return V


# ═══════════════════════════════════════════════════════════════════
#  ALGORITMO SPLIT-STEP DE SEGUNDO ORDEN (STRANG SPLITTING)
#
#  e^{-i(T+V)Δt/ħ} ≈ e^{-iVΔt/2ħ} · e^{-iTΔt/ħ} · e^{-iVΔt/2ħ}
#  Error de truncamiento: O(Δt³) por paso → O(Δt²) global
# ═══════════════════════════════════════════════════════════════════

def propagator_kinetic(phi_k, k_arr, dt):
    """Propagador cinético exacto en espacio k."""
    phase = np.exp(-1j * hbar * k_arr**2 / (2*m) * dt)
    return phi_k * phase


def propagator_potential(psi, V, dt):
    """Propagador de potencial en espacio x (exacto si V no depende de t)."""
    phase = np.exp(-1j * V / hbar * dt)
    return psi * phase


def split_step_full(psi, V, k_arr, dt):
    """
    Un paso completo del metodo split-step de segundo orden.
    Para V=0 equivale exactamente al metodo original.
    """
    # Medio paso de potencial
    psi = propagator_potential(psi, V, dt/2)
    # Paso completo cinético en espacio k
    phi_k = np.fft.fft(psi)
    phi_k = propagator_kinetic(phi_k, k_arr, dt)
    psi = np.fft.ifft(phi_k)
    # Medio paso de potencial
    psi = propagator_potential(psi, V, dt/2)
    return psi


# ═══════════════════════════════════════════════════════════════════
#  SIMULACIÓN 1D COMPLETA CON REGISTRO DE OBSERVABLES
# ═══════════════════════════════════════════════════════════════════

def simulate_1D(V_func, label="Libre", save_every=10,
                x0_init=-12.0, k0_init=5.0, sigma_init=1.0,
                T_sim=8.0, dt_sim=0.01, verbose=True):
    """
    Ejecuta la simulación completa con el potencial V_func dado.
    Retorna diccionario con observables y estados en tiempos seleccionados.
    """
    n_sim   = int(T_sim / dt_sim)
    t_vec   = np.arange(n_sim + 1) * dt_sim
    V       = V_func(x)

    psi     = psi0_gauss(x, x0=x0_init, sigma0=sigma_init, k0=k0_init)
    psi     = normalize(psi, dx)

    # Energía cinética analítica inicial (referencia)
    E_kin_0 = (hbar**2 / (2*m)) * (k0_init**2 + 1/(2*sigma_init**2))

    # Almacenamiento de observables
    norma_t = []
    xm_t    = []
    sig_t   = []
    Ekin_t  = []
    Epot_t  = []

    # Estados completos en tiempos seleccionados
    states  = {}
    t_saved = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Simulando: {label}")
        print(f"  E_inicial = {E_kin_0:.4f}, k0 = {k0_init}, σ₀ = {sigma_init}")
        print(f"  Pasos: {n_sim},  dt = {dt_sim},  T = {T_sim}")
        print(f"{'='*60}")

    for step in range(n_sim + 1):
        t_now = step * dt_sim

        # Registrar observables
        rho   = np.abs(psi)**2
        norm  = np.sum(rho) * dx
        xm    = np.sum(x * rho) * dx / norm if norm > 0 else 0
        x2m   = np.sum(x**2 * rho) * dx / norm if norm > 0 else 0
        sig   = np.sqrt(max(x2m - xm**2, 0.0))
        Ekin  = energy_kinetic(psi, k_arr, dx)
        Epot  = np.real(np.sum(rho * V) * dx)

        norma_t.append(norm)
        xm_t.append(xm)
        sig_t.append(sig)
        Ekin_t.append(Ekin)
        Epot_t.append(Epot)

        # Guardar estado completo periódicamente
        if step % save_every == 0:
            states[t_now] = np.copy(psi)
            t_saved.append(t_now)

        if step < n_sim:
            psi = split_step_full(psi, V, k_arr, dt_sim)

    if verbose:
        print(f"  ✓ Completado")
        print(f"  Norma final: {norma_t[-1]:.8f}")
        print(f"  Posición final ⟨x⟩: {xm_t[-1]:.4f}")

    return {
        'label': label,
        'V': V,
        't': np.array(t_vec),
        'norma': np.array(norma_t),
        'xm': np.array(xm_t),
        'sigma': np.array(sig_t),
        'Ekin': np.array(Ekin_t),
        'Epot': np.array(Epot_t),
        'Etot': np.array(Ekin_t) + np.array(Epot_t),
        'states': states,
        't_saved': np.array(t_saved),
        'E0': E_kin_0,
    }


# ═══════════════════════════════════════════════════════════════════
#  CÁLCULO DE COEFICIENTES DE TRANSMISIÓN Y REFLEXIÓN
# ═══════════════════════════════════════════════════════════════════

def transmission_reflection(psi, x, dx, x_barrier=0.0):
    """
    Calcula probabilidades de transmisión T y reflexión R
    integrando |ψ|² a ambos lados de la barrera.
    """
    rho = np.abs(psi)**2
    mask_R = x < x_barrier
    mask_T = x >= x_barrier
    R = np.sum(rho[mask_R]) * dx
    T = np.sum(rho[mask_T]) * dx
    total = R + T
    return T/total if total > 0 else 0, R/total if total > 0 else 0


# ═══════════════════════════════════════════════════════════════════
#  EXTENSIÓN A 2D
# ═══════════════════════════════════════════════════════════════════

def simulate_2D(V2D_func=None, Nx=128, Ny=128, Lx=20.0, Ly=20.0,
                x0_2d=-8.0, y0_2d=0.0, kx0=4.0, ky0=0.0,
                sx=1.0, sy=1.0, dt_2d=0.02, T_2d=5.0,
                save_times=None, label_2d="Libre 2D", verbose=True):
    """
    Simulación 2D del paquete gaussiano con potencial arbitrario V(x,y).
    Usa split-step 2D: FFT2 en ambas dimensiones.
    """
    xx = np.linspace(-Lx, Lx, Nx, endpoint=False)
    yy = np.linspace(-Ly, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(xx, yy, indexing='ij')
    dxx = xx[1] - xx[0]
    dyy = yy[1] - yy[0]

    # Grillas de momentos 2D
    kx = 2*pi * np.fft.fftfreq(Nx, d=dxx)
    ky = 2*pi * np.fft.fftfreq(Ny, d=dyy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Estado inicial 2D
    Norm2D = (pi * sx * sy)**(-0.5)
    psi2D = (Norm2D *
             np.exp(-(X - x0_2d)**2 / (2*sx**2)) *
             np.exp(-(Y - y0_2d)**2 / (2*sy**2)) *
             np.exp(1j*(kx0*X + ky0*Y)))

    # Normalizar
    norm0 = np.sqrt(np.sum(np.abs(psi2D)**2) * dxx * dyy)
    psi2D /= norm0

    # Potencial 2D
    if V2D_func is None:
        V2D = np.zeros_like(X)
    else:
        V2D = V2D_func(X, Y)

    # Propagadores
    phase_kin_2d = np.exp(-1j * hbar * (KX**2 + KY**2) / (2*m) * dt_2d)
    phase_pot_half = np.exp(-1j * V2D / hbar * dt_2d / 2)

    n_2d = int(T_2d / dt_2d)
    if save_times is None:
        save_times = np.linspace(0, T_2d, 6)

    states_2d = {}
    norma_2d  = []
    xm_2d     = []
    ym_2d     = []
    t_2d_arr  = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Simulación 2D: {label_2d}")
        print(f"  Grid: {Nx}×{Ny}, dominio: [{-Lx},{Lx}]×[{-Ly},{Ly}]")
        print(f"{'='*60}")

    for step in range(n_2d + 1):
        t_now = step * dt_2d
        rho2d = np.abs(psi2D)**2
        norm  = np.sum(rho2d) * dxx * dyy
        norma_2d.append(norm)
        xm_2d.append(np.sum(X * rho2d) * dxx * dyy / norm)
        ym_2d.append(np.sum(Y * rho2d) * dxx * dyy / norm)
        t_2d_arr.append(t_now)

        # Guardar en tiempos seleccionados
        for ts in save_times:
            if abs(t_now - ts) < dt_2d/2 and ts not in states_2d:
                states_2d[ts] = np.copy(psi2D)

        if step < n_2d:
            psi2D = phase_pot_half * psi2D
            phi2d = np.fft.fft2(psi2D)
            phi2d *= phase_kin_2d
            psi2D = np.fft.ifft2(phi2d)
            psi2D = phase_pot_half * psi2D

    if verbose:
        print(f"  ✓ Completado. Norma final: {norma_2d[-1]:.8f}")

    return {
        'label': label_2d,
        'xx': xx, 'yy': yy, 'X': X, 'Y': Y,
        'V2D': V2D,
        't': np.array(t_2d_arr),
        'norma': np.array(norma_2d),
        'xm': np.array(xm_2d),
        'ym': np.array(ym_2d),
        'states': states_2d,
        'save_times': save_times,
        'dxx': dxx, 'dyy': dyy,
    }


# ═══════════════════════════════════════════════════════════════════
#  POTENCIALES 2D
# ═══════════════════════════════════════════════════════════════════

def V2D_free(X, Y):
    return np.zeros_like(X)


def V2D_barrier(X, Y, V0=40.0, x_c=0.0, width=1.0):
    """Barrera vertical en x=x_c."""
    V = np.zeros_like(X)
    V[np.abs(X - x_c) <= width/2] = V0
    return V


def V2D_double_slit(X, Y, V0=200.0, x_c=0.0, width=0.5,
                    slit_sep=4.0, slit_width=1.5):
    """
    Doble ranura: pared en x=x_c con dos aperturas en y.
    Modelo clásico de Young en mecánica cuántica.
    """
    V = np.zeros_like(X)
    wall = np.abs(X - x_c) <= width/2
    slit1 = np.abs(Y - slit_sep/2) <= slit_width/2
    slit2 = np.abs(Y + slit_sep/2) <= slit_width/2
    V[wall & ~slit1 & ~slit2] = V0
    return V


def V2D_harmonic(X, Y, omega=0.5):
    """Pozo armónico 2D isotrópico."""
    return 0.5 * m * omega**2 * (X**2 + Y**2)


def V2D_circular_barrier(X, Y, V0=30.0, R=5.0, width=0.5):
    """Barrera circular (dispersor cuántico)."""
    r = np.sqrt(X**2 + Y**2)
    V = np.zeros_like(X)
    V[np.abs(r - R) <= width/2] = V0
    return V


# ═══════════════════════════════════════════════════════════════════
#  GENERACIÓN DE FIGURAS
# ═══════════════════════════════════════════════════════════════════

def plot_1D_comparison(results_list, filename="fig_1D_comparison.png"):
    """
    Compara evolución de |ψ|² para distintos potenciales en un solo grid.
    """
    n_pot = len(results_list)
    t_plot = [0.0, 2.0, 4.0, 6.0, 8.0]

    fig = plt.figure(figsize=(18, 4*n_pot))
    gs  = gridspec.GridSpec(n_pot, len(t_plot)+1,
                            width_ratios=[1]*len(t_plot)+[0.8],
                            wspace=0.35, hspace=0.45)

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(t_plot)))

    for row, res in enumerate(results_list):
        V = res['V']
        # Normalizar potencial para graficar
        Vmax = np.max(np.abs(V))
        Vplot = V / Vmax * 0.5 if Vmax > 0 else V

        for col, t_target in enumerate(t_plot):
            ax = fig.add_subplot(gs[row, col])
            # Buscar estado más cercano
            t_keys = list(res['states'].keys())
            t_near = min(t_keys, key=lambda t: abs(t - t_target))
            psi_t  = res['states'][t_near]
            rho    = np.abs(psi_t)**2

            ax.fill_between(x, rho, alpha=0.5, color=colors[col])
            ax.plot(x, rho, color=colors[col], lw=1.2)
            if Vmax > 0:
                ax.fill_between(x, Vplot, 0, alpha=0.25,
                                color='gray', label='V(x)')
                ax.plot(x, Vplot, 'k--', lw=0.8, alpha=0.6)
            ax.set_xlim(-L, L)
            ax.set_ylim(-0.02, max(np.max(rho)*1.2, 0.1))
            ax.set_xlabel('x', fontsize=9)
            if col == 0:
                ax.set_ylabel('|ψ|²', fontsize=9)
            ax.set_title(f't = {t_near:.1f}', fontsize=9)
            ax.tick_params(labelsize=8)
            if col == 0 and row == 0:
                ax.set_title(f't = {t_near:.1f}\n{res["label"]}', fontsize=9)
            elif col == 0:
                ax.set_title(f't = {t_near:.1f}', fontsize=9)
                ax.set_ylabel(f'{res["label"]}\n|ψ|²', fontsize=9)

        # Panel de observables (último columna)
        ax_obs = fig.add_subplot(gs[row, -1])
        t_arr  = res['t']
        ax_obs.plot(t_arr, res['norma'], label='Norma', lw=1.5, color='steelblue')
        ax_obs.plot(t_arr, res['Ekin'] / res['E0'], label='E_kin/E₀',
                    lw=1.5, color='darkorange', ls='--')
        ax_obs.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax_obs.set_xlabel('t', fontsize=9)
        ax_obs.set_title('Conservación', fontsize=9)
        ax_obs.legend(fontsize=7, loc='lower left')
        ax_obs.set_ylim(0.9, 1.1)
        ax_obs.tick_params(labelsize=8)
        ax_obs.grid(True, alpha=0.3)

    fig.suptitle('Comparación de potenciales — Evolución de |ψ(x,t)|²',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename}")


def plot_barrier_detail(res_barrier, filename="fig_barrier_detail.png"):
    """Análisis detallado de la barrera: transmisión, reflexión, espectro."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Análisis detallado — Barrera de potencial', fontsize=12, fontweight='bold')

    # Transmisión y reflexión en función del tiempo
    ax = axes[0, 0]
    T_list, R_list = [], []
    for t_k in res_barrier['t_saved']:
        psi_t = res_barrier['states'][t_k]
        T_c, R_c = transmission_reflection(psi_t, x, dx)
        T_list.append(T_c)
        R_list.append(R_c)
    ax.plot(res_barrier['t_saved'], T_list, label='Transmisión T', color='teal', lw=2)
    ax.plot(res_barrier['t_saved'], R_list, label='Reflexión R', color='crimson', lw=2)
    ax.set_xlabel('t'); ax.set_ylabel('Probabilidad')
    ax.set_title('Coeficientes T y R vs tiempo'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Posición media
    ax = axes[0, 1]
    ax.plot(res_barrier['t'], res_barrier['xm'], color='navy', lw=1.5)
    ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('t'); ax.set_ylabel('⟨x⟩')
    ax.set_title('Posición media'); ax.grid(alpha=0.3)

    # Ancho σ(t)
    ax = axes[0, 2]
    ax.plot(res_barrier['t'], res_barrier['sigma'], color='purple', lw=1.5)
    ax.set_xlabel('t'); ax.set_ylabel('σ(t)')
    ax.set_title('Ancho del paquete'); ax.grid(alpha=0.3)

    # Densidad en t=0, t=4, t=8
    t_plots = [0.0, 3.0, 6.0, 8.0]
    colors_p = ['royalblue', 'darkorange', 'forestgreen', 'crimson']
    ax = axes[1, 0]
    V = res_barrier['V']
    ax.fill_between(x, V/np.max(V)*0.4, 0, alpha=0.2, color='gray')
    ax.plot(x, V/np.max(V)*0.4, 'k--', lw=0.8)
    for t_target, col in zip(t_plots, colors_p):
        t_keys = list(res_barrier['states'].keys())
        t_near = min(t_keys, key=lambda t: abs(t - t_target))
        rho = np.abs(res_barrier['states'][t_near])**2
        ax.plot(x, rho, color=col, lw=1.5, label=f't={t_near:.1f}', alpha=0.85)
    ax.set_xlabel('x'); ax.set_ylabel('|ψ|²')
    ax.set_title('Densidad de probabilidad'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(-L, L)

    # Espectro de momentos (constante)
    ax = axes[1, 1]
    t_keys = list(res_barrier['states'].keys())
    for t_target, col in zip([0.0, 4.0, 8.0], ['royalblue', 'darkorange', 'crimson']):
        t_near = min(t_keys, key=lambda t: abs(t - t_target))
        psi_t = res_barrier['states'][t_near]
        phi_k = np.fft.fft(psi_t)
        spec  = np.abs(np.fft.fftshift(phi_k))**2
        k_s   = np.fft.fftshift(k_arr)
        ax.plot(k_s, spec/np.max(spec), color=col, lw=1.2,
                label=f't={t_near:.1f}', alpha=0.8)
    ax.set_xlim(-15, 15); ax.set_xlabel('k'); ax.set_ylabel('|φ(k)|² (norm.)')
    ax.set_title('Espectro de momentos'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Energía total
    ax = axes[1, 2]
    ax.plot(res_barrier['t'], res_barrier['Ekin'],  label='E_cin', lw=1.5, color='darkorange')
    ax.plot(res_barrier['t'], res_barrier['Epot'],  label='E_pot', lw=1.5, color='royalblue')
    ax.plot(res_barrier['t'], res_barrier['Etot'],  label='E_tot', lw=2,   color='black')
    ax.set_xlabel('t'); ax.set_ylabel('Energía')
    ax.set_title('Energías cinética, potencial y total'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename}")


def plot_2D_results(res2d, filename="fig_2D_results.png"):
    """Panel multipanel para la simulación 2D."""
    save_times = res2d['save_times']
    n_t = min(len(save_times), 6)
    t_plot = save_times[:n_t]

    fig = plt.figure(figsize=(4*n_t, 8))
    gs  = gridspec.GridSpec(2, n_t, hspace=0.4, wspace=0.3)

    cmap = 'inferno'
    xx, yy = res2d['xx'], res2d['yy']
    V2D = res2d['V2D']
    Vmax = np.max(V2D)

    for col, t_target in enumerate(t_plot):
        t_keys = list(res2d['states'].keys())
        t_near = min(t_keys, key=lambda t: abs(t - t_target))
        psi_t  = res2d['states'].get(t_near, list(res2d['states'].values())[col])
        rho2d  = np.abs(psi_t)**2

        # Fila superior: densidad 2D
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(xx, yy, rho2d.T, cmap=cmap, shading='auto')
        if Vmax > 0:
            ax.contour(xx, yy, V2D.T, levels=[Vmax*0.5],
                       colors='white', linewidths=0.8, alpha=0.7)
        ax.set_aspect('equal')
        ax.set_title(f't = {t_near:.1f}', fontsize=9)
        ax.set_xlabel('x', fontsize=8); ax.set_ylabel('y', fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Fila inferior: perfil en y=0 (corte horizontal)
        ax2 = fig.add_subplot(gs[1, col])
        iy0 = len(yy)//2
        ax2.fill_between(xx, rho2d[:, iy0], alpha=0.5, color='steelblue')
        ax2.plot(xx, rho2d[:, iy0], color='steelblue', lw=1.2)
        ax2.set_xlabel('x', fontsize=8)
        ax2.set_title(f'Corte y=0, t={t_near:.1f}', fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Simulación 2D — {res2d["label"]}\nDensidad de probabilidad |ψ(x,y,t)|²',
                 fontsize=12, fontweight='bold')
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename}")


def plot_double_slit_2D(res2d, filename="fig_double_slit.png"):
    """Visualización especial para el experimento de doble ranura."""
    t_keys = sorted(res2d['states'].keys())
    t_final = t_keys[-1]
    psi_final = res2d['states'][t_final]
    rho_final = np.abs(psi_final)**2

    xx, yy = res2d['xx'], res2d['yy']
    V2D = res2d['V2D']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Doble ranura cuántica — Experimento de Young', fontsize=13, fontweight='bold')

    # Panel 1: Densidad 2D final
    ax = axes[0]
    im = ax.pcolormesh(xx, yy, rho_final.T, cmap='hot', shading='auto')
    ax.contour(xx, yy, V2D.T, levels=[np.max(V2D)*0.3],
               colors='cyan', linewidths=1.0, alpha=0.8)
    ax.set_aspect('equal')
    ax.set_title(f'Densidad |ψ|² en t={t_final:.1f}', fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

    # Panel 2: Patrón de interferencia en x=L/2 (pantalla)
    ax = axes[1]
    ix_screen = np.argmin(np.abs(xx - xx[-1]*0.6))
    pattern = rho_final[ix_screen, :]
    ax.fill_betweenx(yy, pattern, alpha=0.5, color='gold')
    ax.plot(pattern, yy, color='darkorange', lw=2)
    ax.set_xlabel('Intensidad'); ax.set_ylabel('y')
    ax.set_title('Patrón de interferencia\n(corte transversal)', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Evolución de norma en 2D
    ax = axes[2]
    ax.plot(res2d['t'], res2d['norma'], color='steelblue', lw=2)
    ax.axhline(1.0, color='gray', lw=1, ls='--')
    ax.set_xlabel('t'); ax.set_ylabel('Norma')
    ax.set_title('Conservación de probabilidad'); ax.grid(alpha=0.3)
    ax.set_ylim(0.9, 1.05)

    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename}")


def plot_observables_comparison(results_list, filename="fig_observables_compare.png"):
    """Compara observables temporales entre distintos potenciales."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Observables comparados entre potenciales', fontsize=12, fontweight='bold')

    colors = ['royalblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
    styles = ['-', '--', '-.', ':', '-']

    for i, res in enumerate(results_list):
        col = colors[i % len(colors)]
        sty = styles[i % len(styles)]
        lbl = res['label']
        t   = res['t']

        axes[0,0].plot(t, res['xm'],    color=col, ls=sty, lw=1.8, label=lbl)
        axes[0,1].plot(t, res['sigma'], color=col, ls=sty, lw=1.8, label=lbl)
        axes[1,0].plot(t, res['norma'], color=col, ls=sty, lw=1.8, label=lbl)
        axes[1,1].plot(t, res['Etot'],  color=col, ls=sty, lw=1.8, label=lbl)

    titles = ['Posición media ⟨x⟩(t)', 'Ancho σ(t)',
              'Norma ∫|ψ|²dx', 'Energía total E(t)']
    ylabels = ['⟨x⟩', 'σ(t)', 'Norma', 'E_total']

    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('t', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[1,0].axhline(1.0, color='gray', lw=0.8, ls=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename}")


def animate_1D(res, filename="animation_1D.gif", fps=15):
    """Genera GIF animado para una simulación 1D."""
    t_keys = sorted(res['states'].keys())
    V = res['V']
    Vmax = np.max(np.abs(V)) if np.max(np.abs(V)) > 0 else 1.0
    rho_max = max(np.max(np.abs(res['states'][t])**2) for t in t_keys) * 1.15

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(-L, L); ax.set_ylim(0, rho_max)
    ax.set_xlabel('x'); ax.set_ylabel('|ψ(x,t)|²')

    if Vmax > 1e-10:
        Vp = V / Vmax * rho_max * 0.4
        ax.fill_between(x, Vp, 0, alpha=0.2, color='gray')
        ax.plot(x, Vp, 'k--', lw=0.8, alpha=0.5, label='V(x)')

    fill = ax.fill_between(x, np.zeros_like(x), alpha=0.4, color='royalblue')
    line, = ax.plot(x, np.zeros_like(x), color='royalblue', lw=1.5)
    time_text = ax.text(0.02, 0.93, '', transform=ax.transAxes, fontsize=10)
    ax.set_title(f'Paquete de ondas — {res["label"]}')

    def update(frame):
        t_k = t_keys[frame]
        rho = np.abs(res['states'][t_k])**2
        line.set_ydata(rho)
        time_text.set_text(f't = {t_k:.2f}')
        return line, time_text

    ani = FuncAnimation(fig, update, frames=len(t_keys), blit=False, interval=60)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  ✓ {filename}")


# ═══════════════════════════════════════════════════════════════════
#  PROGRAMA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 65)
    print("  SIMULACIÓN EXTENDIDA — PAQUETE DE ONDAS GAUSSIANO")
    print("  Con potenciales V(x)≠0 y extensión a 2D")
    print("=" * 65)

    # ── BLOQUE 1: SIMULACIONES 1D ──────────────────────────────────

    print("\n[1] Simulaciones 1D con distintos potenciales")

    # Parámetros comunes para todas las simulaciones 1D
    common = dict(x0_init=-12.0, k0_init=5.0, sigma_init=1.0,
                  T_sim=8.0, dt_sim=0.01, save_every=20)

    # Energía cinética inicial (para comparar con la barrera)
    E0 = (hbar**2 / (2*m)) * (common['k0_init']**2 + 1/(2*common['sigma_init']**2))
    print(f"\n  Energía cinética inicial: E₀ = {E0:.3f}")
    print(f"  Barrera V₀ = 25 < E₀ → transmisión parcial clásica")
    print(f"  Barrera V₀ = {E0:.1f} ≈ E₀ → régimen cuántico de efecto túnel")

    results_1D = []

    # 1. Partícula libre (referencia)
    res_free = simulate_1D(V_free, label="Libre (V=0)", **common)
    results_1D.append(res_free)

    # 2. Barrera menor que E₀ (transmisión parcial + reflexión)
    res_bar_low = simulate_1D(
        lambda x: V_barrier(x, V0=12.0, x_c=0.0, width=2.0),
        label="Barrera V₀=12 < E₀", **common)
    results_1D.append(res_bar_low)

    # 3. Barrera mayor que E₀ (efecto túnel dominante)
    res_bar_high = simulate_1D(
        lambda x: V_barrier(x, V0=15.0, x_c=0.0, width=2.0),
        label="Barrera V₀=15 ≈ E₀ (túnel)", **common)
    results_1D.append(res_bar_high)

    # 4. Pozo armónico (estados ligados)
    res_harm = simulate_1D(
        lambda x: V_harmonic(x, omega=1.5, x_c=5.0),
        label="Pozo armónico ω=1.5",
        x0_init=-8.0, k0_init=3.0, sigma_init=1.0,
        T_sim=8.0, dt_sim=0.01, save_every=20)
    results_1D.append(res_harm)

    # 5. Escalón de potencial
    res_step = simulate_1D(
        lambda x: V_step(x, V0=10.0, x_c=0.0),
        label="Escalón V₀=10", **common)
    results_1D.append(res_step)

    # ── BLOQUE 2: FIGURAS 1D ──────────────────────────────────────

    print("\n[2] Generando figuras 1D...")
    plot_1D_comparison(results_1D,
                       filename="fig_1D_comparison.png")
    plot_barrier_detail(res_bar_high,
                        filename="fig_barrier_detail.png")
    plot_observables_comparison(results_1D,
                                filename="fig_observables_compare.png")
    animate_1D(res_bar_high,
               filename="animation_barrier.gif", fps=12)

    # ── BLOQUE 3: TABLA TRANSMISIÓN ────────────────────────────────

    print("\n[3] Coeficientes de transmisión/reflexión finales")
    print(f"{'Potencial':<30} {'T':>8} {'R':>8} {'T+R':>8}")
    print("-" * 56)
    for res in results_1D:
        t_keys = list(res['states'].keys())
        t_last = max(t_keys)
        psi_last = res['states'][t_last]
        T_c, R_c = transmission_reflection(psi_last, x, dx)
        print(f"  {res['label']:<28} {T_c:>8.4f} {R_c:>8.4f} {T_c+R_c:>8.4f}")

    # ── BLOQUE 4: SIMULACIONES 2D ──────────────────────────────────

    print("\n[4] Simulaciones 2D")

    save_t_2d = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    # 2D libre
    res2d_free = simulate_2D(
        V2D_func=None,
        Nx=128, Ny=128, Lx=18.0, Ly=18.0,
        x0_2d=-8.0, y0_2d=0.0, kx0=4.0, ky0=0.0,
        sx=1.2, sy=1.2, dt_2d=0.02, T_2d=5.0,
        save_times=save_t_2d,
        label_2d="Libre 2D")

    # 2D con barrera vertical
    res2d_bar = simulate_2D(
        V2D_func=lambda X,Y: V2D_barrier(X, Y, V0=40.0, x_c=0.0, width=0.8),
        Nx=128, Ny=128, Lx=18.0, Ly=18.0,
        x0_2d=-8.0, y0_2d=0.0, kx0=4.0, ky0=0.0,
        sx=1.2, sy=1.2, dt_2d=0.02, T_2d=5.0,
        save_times=save_t_2d,
        label_2d="Barrera vertical 2D")

    # 2D doble ranura (experimento de Young cuántico)
    res2d_ds = simulate_2D(
        V2D_func=lambda X,Y: V2D_double_slit(
            X, Y, V0=300.0, x_c=0.0, width=0.6,
            slit_sep=5.0, slit_width=1.2),
        Nx=128, Ny=128, Lx=18.0, Ly=18.0,
        x0_2d=-8.0, y0_2d=0.0, kx0=4.0, ky0=0.0,
        sx=1.0, sy=3.0, dt_2d=0.02, T_2d=5.0,
        save_times=save_t_2d,
        label_2d="Doble ranura 2D (Young)")

    # ── BLOQUE 5: FIGURAS 2D ──────────────────────────────────────

    print("\n[5] Generando figuras 2D...")
    plot_2D_results(res2d_free,  filename="fig_2D_free.png")
    plot_2D_results(res2d_bar,   filename="fig_2D_barrier.png")
    plot_double_slit_2D(res2d_ds, filename="fig_double_slit.png")

    # ── RESUMEN FINAL ──────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("  ARCHIVOS GENERADOS")
    print("=" * 65)
    files = [
        "fig_1D_comparison.png     — Comparación de potenciales 1D",
        "fig_barrier_detail.png    — Análisis barrera: T, R, energías",
        "fig_observables_compare.png — Observables vs tiempo",
        "animation_barrier.gif     — Animación barrera (efecto túnel)",
        "fig_2D_free.png           — Paquete libre en 2D",
        "fig_2D_barrier.png        — Barrera vertical en 2D",
        "fig_double_slit.png       — Doble ranura cuántica (Young)",
    ]
    for f in files:
        print(f"  ✓  {f}")

    print("\n  Simulación completada exitosamente.")
    print("=" * 65)