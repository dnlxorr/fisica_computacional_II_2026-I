#!/usr/bin/env python3
# =============================================================================
# MOVIMIENTO BROWNIANO: ECUACIÓN DE LANGEVIN Y MÉTODOS NUMÉRICOS
# =============================================================================
# Proyecto de Física Computacional y Estadística
#
# UNIDADES REDUCIDAS (adimensionales):
#   Se fija kB = 1. Los parámetros m, γ, T, v0 son adimensionales.
#   Esto es estándar en física estadística computacional (igual que LAMMPS,
#   GROMACS etc. con "reduced units"). La física es idéntica; simplemente
#   se elige la escala de energía kB·T como unidad de energía.
#
# Dependencias: numpy, scipy, matplotlib
#   pip install numpy scipy matplotlib
#
# Uso:
#   python brownian_simulation.py
#
# Todas las figuras se guardan en ./figuras/ como PDF y PNG.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d

# -----------------------------------------------------------------------------
# 0. CONFIGURACIÓN GLOBAL
# -----------------------------------------------------------------------------
os.makedirs("figuras", exist_ok=True)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ---- Unidades reducidas: kB = 1 ----
kB    = 1.0     # constante de Boltzmann [unidades reducidas]

# Parámetros físicos
m     = 1.0     # masa                   [u.r.]
gamma = 1.0     # coeficiente fricción   [u.r.]
T     = 1.0     # temperatura            [u.r.]  (kB·T = 1)
v0    = 2.0     # velocidad inicial      [u.r.]
x0    = 0.0     # posición inicial       [u.r.]

# Parámetros derivados
beta    = gamma / m                         # amortiguamiento  [1/t]
tau_rel = m / gamma                         # tiempo relajación
D       = kB * T / gamma                    # coeficiente difusión
sigma_v = np.sqrt(2.0 * gamma * kB * T) / m  # amplitud ruido velocidad

print("=" * 62)
print("  MOVIMIENTO BROWNIANO — PARÁMETROS (UNIDADES REDUCIDAS kB=1)")
print("=" * 62)
print(f"  m={m},  γ={gamma},  T={T},  kB={kB}")
print(f"  β = γ/m     = {beta:.4f}  τ = m/γ = {tau_rel:.4f}")
print(f"  D = kBT/γ   = {D:.6f}")
print(f"  σ_v         = {sigma_v:.6f}")
print("=" * 62)

# Parámetros de integración
t_final = 10.0
dt      = 0.01
N_steps = int(t_final / dt)
t_array = np.linspace(0.0, t_final, N_steps + 1)

# Ensamble
N_traj = 2000

# =============================================================================
# 1. SOLUCIÓN ANALÍTICA (DETERMINISTA)
# =============================================================================
def solucion_analitica(t, v0=v0, x0=x0, beta=beta):
    """
    Solución exacta de dv/dt = -β v:
      v(t) = v0 · exp(−β t)
      x(t) = x0 + (v0/β)(1 − exp(−β t))
    """
    v_a = v0 * np.exp(-beta * t)
    x_a = x0 + (v0 / beta) * (1.0 - np.exp(-beta * t))
    return v_a, x_a

v_anal, x_anal = solucion_analitica(t_array)
print(f"\n[1] Solución analítica: {len(t_array)} puntos.")

# =============================================================================
# 2. EULER DETERMINISTA
# =============================================================================
def euler_determinista(t_array, v0=v0, x0=x0, beta=beta):
    """
    Euler explícito: v_{n+1} = v_n(1 − β Δt),  x_{n+1} = x_n + Δt v_n
    """
    N  = len(t_array)
    dt = t_array[1] - t_array[0]
    v  = np.zeros(N); x = np.zeros(N)
    v[0] = v0; x[0] = x0
    for n in range(N-1):
        v[n+1] = v[n] * (1.0 - beta * dt)
        x[n+1] = x[n] + dt * v[n]
    return v, x

v_euler, x_euler = euler_determinista(t_array)
print("[2] Euler determinista completado.")

# =============================================================================
# 3. RUNGE-KUTTA 4 DETERMINISTA
# =============================================================================
def rk4_determinista(t_array, v0=v0, x0=x0, beta=beta):
    """
    RK4 clásico:
      k1 = f(vn)
      k2 = f(vn + dt/2 · k1)
      k3 = f(vn + dt/2 · k2)
      k4 = f(vn + dt   · k3)
      v_{n+1} = vn + dt/6 · (k1 + 2k2 + 2k3 + k4)
    """
    N  = len(t_array)
    dt = t_array[1] - t_array[0]
    v  = np.zeros(N); x = np.zeros(N)
    v[0] = v0; x[0] = x0
    f = lambda vv: -beta * vv
    for n in range(N-1):
        vn = v[n]
        k1 = f(vn)
        k2 = f(vn + 0.5*dt*k1)
        k3 = f(vn + 0.5*dt*k2)
        k4 = f(vn +     dt*k3)
        v[n+1] = vn + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x[n+1] = x[n] + dt*vn
    return v, x

v_rk4, x_rk4 = rk4_determinista(t_array)
print("[3] RK4 determinista completado.")

err_euler_abs = np.abs(v_euler - v_anal)
err_rk4_abs   = np.abs(v_rk4  - v_anal)
print(f"[4] Errores (dt={dt}):  Euler={err_euler_abs.max():.3e}  RK4={err_rk4_abs.max():.3e}")

# =============================================================================
# 5. CONVERGENCIA — BARRIDO DE Δt
# =============================================================================
dt_values     = np.logspace(-3, 0, 22)
max_err_euler = np.zeros(len(dt_values))
max_err_rk4   = np.zeros(len(dt_values))

for i, dti in enumerate(dt_values):
    Ni = max(int(t_final / dti), 1)
    ti = np.linspace(0.0, Ni*dti, Ni+1)
    # Euler
    vi_e = np.zeros(Ni+1); vi_e[0] = v0
    for n in range(Ni): vi_e[n+1] = vi_e[n]*(1 - beta*dti)
    # RK4
    vi_r = np.zeros(Ni+1); vi_r[0] = v0
    for n in range(Ni):
        vn=vi_r[n]; k1=-beta*vn; k2=-beta*(vn+.5*dti*k1)
        k3=-beta*(vn+.5*dti*k2); k4=-beta*(vn+dti*k3)
        vi_r[n+1]=vn+(dti/6)*(k1+2*k2+2*k3+k4)
    v_ex = v0*np.exp(-beta*ti)
    max_err_euler[i] = np.max(np.abs(vi_e - v_ex))
    max_err_rk4[i]   = np.max(np.abs(vi_r - v_ex))

mask_fit = dt_values < 0.3
sl_eu,_,_,_,_ = stats.linregress(np.log(dt_values[mask_fit]),
                                  np.log(max_err_euler[mask_fit]))
sl_rk,_,_,_,_ = stats.linregress(np.log(dt_values[mask_fit]),
                                  np.log(max_err_rk4[mask_fit]))
print(f"[5] Convergencia: Euler pendiente={sl_eu:.3f}  RK4 pendiente={sl_rk:.3f}")

# =============================================================================
# 6. EULER-MARUYAMA — ENSAMBLE VECTORIZADO
# =============================================================================
def euler_maruyama_ensemble(t_array, N_traj, v0=v0, x0=x0,
                             beta=beta, sigma_v=sigma_v, rng=rng):
    """
    EDE de Langevin (Itô):  dv = −β v dt + σ_v dW
    Discretización EM:
      v_{n+1} = v_n − β Δt v_n + σ_v √Δt Z_n   (Z_n ~ N(0,1))
      x_{n+1} = x_n + Δt v_n

    Implementación vectorizada sobre el ensamble (axis=0).
    """
    N_steps = len(t_array) - 1
    dt      = t_array[1] - t_array[0]
    sdt     = np.sqrt(dt)

    v_ens = np.zeros((N_traj, N_steps+1))
    x_ens = np.zeros((N_traj, N_steps+1))
    v_ens[:, 0] = v0
    x_ens[:, 0] = x0

    # Todos los incrementos de una vez → forma (N_traj, N_steps)
    Z = rng.standard_normal((N_traj, N_steps))
    xi_ens = sigma_v * Z / sdt   # fuerza estocástica por unidad de masa

    coef = 1.0 - beta*dt
    for n in range(N_steps):
        v_ens[:, n+1] = coef*v_ens[:, n] + sigma_v*sdt*Z[:, n]
        x_ens[:, n+1] = x_ens[:, n] + dt*v_ens[:, n]

    return v_ens, x_ens, xi_ens

print(f"\n[6] Simulando ensamble EM: {N_traj} trayectorias × {N_steps} pasos...")
v_ens, x_ens, xi_ens = euler_maruyama_ensemble(t_array, N_traj)
print(f"    Listo. Forma arrays: {x_ens.shape}")

# =============================================================================
# 7. MSD
# =============================================================================
displ     = (x_ens - x_ens[:, 0:1])**2
msd_mean  = displ.mean(axis=0)
msd_std   = displ.std(axis=0)
msd_teo   = 2.0 * D * t_array

# Ajuste lineal (excluir t < 0.5 para evitar régimen balístico)
idx_fit    = t_array > 0.5
sl_m,ic_m,r_m,_,se_m = stats.linregress(t_array[idx_fit], msd_mean[idx_fit])
D_num = sl_m / 2.0
print(f"[7] MSD:  D_teo={D:.6f}  D_num={D_num:.6f}  err={abs(D_num-D)/D*100:.2f}%  R²={r_m**2:.6f}")

# =============================================================================
# 8. DISTRIBUCIÓN DE POSICIONES
# =============================================================================
t_star   = 3.0
idx_star = np.argmin(np.abs(t_array - t_star))
x_star   = x_ens[:, idx_star]

# Teórico: gaussiana centrada en posición media del sistema amortiguado
mu_teo    = x0 + (v0/beta)*(1 - np.exp(-beta*t_star))
sigma_teo = np.sqrt(2*D*t_star)

ks_stat, ks_p = stats.kstest(x_star, 'norm', args=(mu_teo, sigma_teo))
print(f"[8] Histograma t*={t_star}:  μ_teo={mu_teo:.4f}  σ_teo={sigma_teo:.4f}")
print(f"    KS stat={ks_stat:.4f}  p={ks_p:.4f}  {'✓ Gaussiana' if ks_p>0.05 else '✗ no gaussiana'}")

# =============================================================================
# 9. ANÁLISIS ESPECTRAL (FFT + WELCH)
# =============================================================================
fs = 1.0 / dt
psd_list = []
for i in range(min(200, N_traj)):
    fi, pi = welch(xi_ens[i], fs=fs, nperseg=256, window='hann')
    psd_list.append(pi)
freqs    = fi
psd_mean_w = np.mean(psd_list, axis=0)
# Teórico espectral (bilateral / 2π): S_ξ = 2γkBT → unilateral Welch ≈ 4γkBT
S_teo_unit = 4.0 * gamma * kB * T

print(f"[9] FFT completado. PSD(f~0) = {np.mean(psd_mean_w[:5]):.4e}  S_teo={S_teo_unit:.4e}")

# =============================================================================
# 10. ANÁLISIS PARAMÉTRICO
# =============================================================================
param_sets = [
    dict(T=0.5, label='T=0.5',  color='#023e8a'),
    dict(T=1.0, label='T=1.0',  color='#0096c7'),
    dict(T=2.0, label='T=2.0',  color='#fb8500'),
    dict(T=4.0, label='T=4.0',  color='#d62828'),
]
rng_p = np.random.default_rng(RNG_SEED+7)
msd_param = {}
for ps in param_sets:
    Ti    = ps['T']
    Di    = kB*Ti/gamma
    sv_i  = np.sqrt(2*gamma*kB*Ti)/m
    Zi    = rng_p.standard_normal((N_traj, N_steps))
    vi    = np.zeros((N_traj, N_steps+1)); xi = np.zeros((N_traj, N_steps+1))
    vi[:,0]=v0; xi[:,0]=x0
    for n in range(N_steps):
        vi[:,n+1] = (1-beta*dt)*vi[:,n] + sv_i*np.sqrt(dt)*Zi[:,n]
        xi[:,n+1] = xi[:,n] + dt*vi[:,n]
    msd_param[ps['label']] = ((xi - xi[:,0:1])**2).mean(axis=0)

# =============================================================================
# 11. FIGURAS
# =============================================================================
plt.rcParams.update({
    'font.family'   : 'serif',
    'font.size'     : 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.8,
    'figure.dpi'    : 150,
    'axes.grid'     : True,
    'grid.alpha'    : 0.3,
})

C_ANAL  = '#1b1b2f'
C_EULER = '#0077b6'
C_RK4   = '#d62828'
C_EM    = '#2d6a4f'
C_GRAY  = '#adb5bd'

print("\n[10] Generando figuras...")

# ---------------------------------------------------------------
# FIG 1: velocidad y posición determinista
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, ya, ye, yr, lbl in zip(axes,
        [v_anal, x_anal], [v_euler, x_euler], [v_rk4, x_rk4],
        [r'Velocidad $v(t)$ [u.r.]', r'Posición $x(t)$ [u.r.]']):
    ax.plot(t_array, ya, color=C_ANAL,  lw=2.2, label='Analítica',   zorder=4)
    ax.plot(t_array, ye, color=C_EULER, lw=1.5, ls='--', label=f'Euler (O(Δt))',  zorder=3)
    ax.plot(t_array, yr, color=C_RK4,  lw=1.5, ls=':',  label='RK4 (O(Δt⁴))', zorder=3)
    ax.set_xlabel(r'$t$ [u.r.]'); ax.set_ylabel(lbl)
    ax.legend()
axes[0].set_title('Velocidad — caso determinista')
axes[1].set_title('Posición — caso determinista')
fig.suptitle(r'Caso determinista: $\dot{v}=-\beta v$,  $\beta=1$,  $\Delta t=0.01$', fontsize=12)
fig.tight_layout()
fig.savefig('figuras/fig_velocidad_det.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_velocidad_det.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_velocidad_det")

# ---------------------------------------------------------------
# FIG 2: error absoluto en tiempo
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.semilogy(t_array, err_euler_abs+1e-20, color=C_EULER, lw=1.8,
            label=f'Euler   max={err_euler_abs.max():.2e}')
ax.semilogy(t_array, err_rk4_abs  +1e-20, color=C_RK4,   lw=1.8,
            label=f'RK4     max={err_rk4_abs.max():.2e}')
ax.set_xlabel(r'$t$ [u.r.]'); ax.set_ylabel(r'$|v_n - v(t_n)|$')
ax.set_title(r'Error absoluto en velocidad — $\Delta t=0.01$')
ax.legend()
fig.tight_layout()
fig.savefig('figuras/fig_error_absoluto.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_error_absoluto.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_error_absoluto")

# ---------------------------------------------------------------
# FIG 3: convergencia log-log
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(dt_values, max_err_euler, 'o-', color=C_EULER, ms=6,
          label=f'Euler  (pendiente={sl_eu:.2f})')
ax.loglog(dt_values, max_err_rk4,   's-', color=C_RK4,  ms=6,
          label=f'RK4    (pendiente={sl_rk:.2f})')
dtr = np.array([dt_values[0], dt_values[-1]])
ax.loglog(dtr, 0.8*dtr**1,    'k--', lw=1.2, alpha=0.55, label=r'$\mathcal{O}(\Delta t)$')
ax.loglog(dtr, 0.05*dtr**4,   'k:',  lw=1.2, alpha=0.55, label=r'$\mathcal{O}(\Delta t^4)$')
ax.set_xlabel(r'$\Delta t$'); ax.set_ylabel('Error máximo absoluto')
ax.set_title('Análisis de convergencia: Euler vs. RK4')
ax.legend(); fig.tight_layout()
fig.savefig('figuras/fig_convergencia.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_convergencia.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_convergencia")

# ---------------------------------------------------------------
# FIG 4: trayectorias brownianas
# ---------------------------------------------------------------
fig, (ax_v, ax_x) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i in range(12):
    ax_v.plot(t_array, v_ens[i], color=C_GRAY, lw=0.7, alpha=0.45)
    ax_x.plot(t_array, x_ens[i], color=C_GRAY, lw=0.7, alpha=0.45)
ax_v.plot(t_array, v_ens[0], color=C_EM, lw=1.8, label='Traj. destacada')
ax_x.plot(t_array, x_ens[0], color=C_EM, lw=1.8)
ax_v.plot(t_array, v_ens.mean(0), color=C_ANAL, lw=2.2, ls='--', label=r'$\langle v\rangle$')
ax_x.plot(t_array, x_ens.mean(0), color=C_ANAL, lw=2.2, ls='--', label=r'$\langle x\rangle$')
ax_v.plot(t_array, v0*np.exp(-beta*t_array), color=C_RK4, lw=1.4, ls=':', label=r'$v_0e^{-\beta t}$')
ax_v.set_ylabel(r'$v(t)$'); ax_v.legend(fontsize=9)
ax_x.set_ylabel(r'$x(t)$'); ax_x.set_xlabel(r'$t$ [u.r.]')
ax_x.legend(fontsize=9)
fig.suptitle(f'Trayectorias Brownianas — Euler-Maruyama  ($N={N_traj}$, $\\Delta t={dt}$)', fontsize=12)
fig.tight_layout()
fig.savefig('figuras/fig_trayectorias.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_trayectorias.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_trayectorias")

# ---------------------------------------------------------------
# FIG 5: MSD
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sem = msd_std / np.sqrt(N_traj)
ax.fill_between(t_array, msd_mean-sem, msd_mean+sem,
                color=C_EM, alpha=0.25, label=r'$\pm$ SEM')
ax.plot(t_array, msd_mean, color=C_EM,   lw=2.0, label=r'$\langle x^2\rangle$ EM')
ax.plot(t_array, msd_teo,  color=C_ANAL, lw=2.0, ls='--',
        label=r'$2Dt$ (teórico)')
mfit = sl_m*t_array + ic_m
ax.plot(t_array[idx_fit], mfit[idx_fit], color=C_RK4, lw=1.6, ls=':',
        label=f'Ajuste lineal: $D_{{\\rm num}}={D_num:.5f}$')
ax.set_xlabel(r'$t$ [u.r.]')
ax.set_ylabel(r'$\langle x^2(t)\rangle$ [u.r.$^2$]')
ax.set_title(r'MSD: verificación de $\langle x^2\rangle = 2Dt$'
             f'\n$D_{{\\rm teo}}={D:.5f}$,  $D_{{\\rm num}}={D_num:.5f}$,  '
             f'error={abs(D_num-D)/D*100:.2f}%')
ax.legend(); fig.tight_layout()
fig.savefig('figuras/fig_msd.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_msd.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_msd")

# ---------------------------------------------------------------
# FIG 6: histograma de posiciones
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
xr = np.linspace(x_star.min(), x_star.max(), 500)
gau_teo = stats.norm.pdf(xr, mu_teo, sigma_teo)
ax.hist(x_star, bins=50, density=True, color=C_EM, alpha=0.65,
        edgecolor='white', lw=0.4, label=f'EM ({N_traj} trajs.)')
ax.plot(xr, gau_teo, color=C_RK4, lw=2.4,
        label=r'$\mathcal{N}(\mu,\,2Dt^*)$ teórica')
ax.axvline(mu_teo, color=C_ANAL, ls='--', lw=1.4, alpha=0.8)
ax.set_xlabel(r'$x(t^*)$ [u.r.]')
ax.set_ylabel('Densidad de probabilidad')
ax.set_title(f'Distribución de posiciones en $t^*={t_star}$ u.r.\n'
             f'KS: stat={ks_stat:.4f},  p={ks_p:.3f}  '
             f'→  {"✓ gaussiana" if ks_p>0.05 else "✗"}')
ax.legend(); fig.tight_layout()
fig.savefig('figuras/fig_histograma.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_histograma.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_histograma")

# ---------------------------------------------------------------
# FIG 7: PSD (FFT / Welch)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
psd_sm = uniform_filter1d(psd_mean_w, size=8)
ax.semilogy(freqs, psd_mean_w, color=C_GRAY, lw=0.9, alpha=0.7, label='PSD (Welch, promedio)')
ax.semilogy(freqs, psd_sm,     color=C_EM,   lw=2.2, label='PSD suavizada')
ax.axhline(S_teo_unit, color=C_RK4, ls='--', lw=1.8,
           label=r'$4\gamma k_B T$ (teórico unilateral)')
ax.set_xlabel(r'Frecuencia $f$ [u.r.$^{-1}$]')
ax.set_ylabel(r'PSD $[u.r.^2 \cdot t]$')
ax.set_title(r'Densidad Espectral de Potencia de $\xi(t)$')
ax.legend(); fig.tight_layout()
fig.savefig('figuras/fig_fft.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_fft.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_fft")

# ---------------------------------------------------------------
# FIG 8: MSD paramétrico (variación de T)
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for ps in param_sets:
    Ti = ps['T']; Di = kB*Ti/gamma
    ax1.plot(t_array, msd_param[ps['label']],
             color=ps['color'], lw=1.8, label=ps['label'])
    ax2.plot(t_array, 2*Di*t_array,
             color=ps['color'], lw=1.8, ls='--', label=f'2·D(T)·t')
ax1.set_xlabel(r'$t$'); ax1.set_ylabel(r'$\langle x^2\rangle$ numérico')
ax1.set_title('MSD numérico — variación de $T$'); ax1.legend()
ax2.set_xlabel(r'$t$'); ax2.set_ylabel(r'$2D(T)t$ teórico')
ax2.set_title('MSD teórico'); ax2.legend()
fig.tight_layout()
fig.savefig('figuras/fig_parametrico_T.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_parametrico_T.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_parametrico_T")

# ---------------------------------------------------------------
# FIG 9: Autocorrelación de velocidad
# ---------------------------------------------------------------
def autocorr(v, max_lag=200):
    """Función de autocorrelación normalizada C(τ) = <v(0)v(τ)>/<v²>."""
    N   = v.shape[1]
    lag = min(max_lag, N//2)
    cv  = np.array([np.mean(v[:,:N-l]*v[:,l:]) for l in range(lag)])
    return cv / cv[0]

print("  Calculando autocorrelación de velocidades...")
max_lag = 300
acv = autocorr(v_ens, max_lag)
tau_lag = np.arange(max_lag) * dt
acv_teo = np.exp(-beta * tau_lag)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(tau_lag, acv,     color=C_EM,   lw=2.0, label=r'$C_v(\tau)$ numérica')
ax.plot(tau_lag, acv_teo, color=C_ANAL, lw=2.0, ls='--',
        label=r'$e^{-\beta\tau}$ teórica')
ax.axhline(0, color='gray', lw=0.8, ls=':')
ax.set_xlabel(r'Retardo $\tau$ [u.r.]')
ax.set_ylabel(r'$C_v(\tau) = \langle v(0)v(\tau)\rangle / \langle v^2\rangle$')
ax.set_title('Autocorrelación de velocidad — Green-Kubo')
ax.legend(); fig.tight_layout()
fig.savefig('figuras/fig_autocorr.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_autocorr.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_autocorr")

# ---------------------------------------------------------------
# FIG 10: Panel resumen general (figura compuesta)
# ---------------------------------------------------------------
fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.36)

# [0,0] Velocidad determinista
ax = fig.add_subplot(gs[0,0])
ax.plot(t_array, v_anal,  color=C_ANAL,  lw=2.0, label='Analítica')
ax.plot(t_array, v_euler, color=C_EULER, lw=1.3, ls='--', label='Euler')
ax.plot(t_array, v_rk4,   color=C_RK4,  lw=1.3, ls=':',  label='RK4')
ax.set_xlabel(r'$t$'); ax.set_ylabel(r'$v(t)$')
ax.set_title('Velocidad det.'); ax.legend(fontsize=7)

# [0,1] Convergencia
ax = fig.add_subplot(gs[0,1])
ax.loglog(dt_values, max_err_euler, 'o-', color=C_EULER, ms=5, label=f'Euler (p={sl_eu:.2f})')
ax.loglog(dt_values, max_err_rk4,   's-', color=C_RK4,  ms=5, label=f'RK4   (p={sl_rk:.2f})')
ax.loglog(dtr, 0.8*dtr,  'k--', lw=1, alpha=0.45, label=r'$O(\Delta t)$')
ax.loglog(dtr, 0.05*dtr**4, 'k:', lw=1, alpha=0.45, label=r'$O(\Delta t^4)$')
ax.set_xlabel(r'$\Delta t$'); ax.set_ylabel('Error máx.')
ax.set_title('Convergencia'); ax.legend(fontsize=6.5)

# [0,2] MSD
ax = fig.add_subplot(gs[0,2])
ax.plot(t_array, msd_mean, color=C_EM,   lw=1.8, label=r'$\langle x^2\rangle$')
ax.plot(t_array, msd_teo,  color=C_ANAL, lw=1.8, ls='--', label=r'$2Dt$')
ax.set_xlabel(r'$t$'); ax.set_ylabel(r'$\langle x^2\rangle$')
ax.set_title('MSD'); ax.legend(fontsize=8)

# [1,0:2] Trayectorias
ax = fig.add_subplot(gs[1,0:2])
for i in range(10):
    ax.plot(t_array, x_ens[i], color=C_GRAY, lw=0.7, alpha=0.4)
ax.plot(t_array, x_ens[0], color=C_EM, lw=1.8)
ax.plot(t_array, x_ens.mean(0), color=C_ANAL, lw=2.2, ls='--', label=r'$\langle x\rangle$')
ax.set_xlabel(r'$t$'); ax.set_ylabel(r'$x(t)$')
ax.set_title('Trayectorias brownianas'); ax.legend(fontsize=8)

# [1,2] Histograma
ax = fig.add_subplot(gs[1,2])
ax.hist(x_star, bins=35, density=True, color=C_EM, alpha=0.6, edgecolor='w', lw=0.3)
ax.plot(xr, gau_teo, color=C_RK4, lw=2.0)
ax.set_xlabel(r'$x(t^*)$'); ax.set_ylabel('PDF')
ax.set_title(f'Dist. posiciones $t^*={t_star}$')

# [2,0:2] Autocorrelación
ax = fig.add_subplot(gs[2,0:2])
ax.plot(tau_lag, acv,     color=C_EM,   lw=2.0, label=r'$C_v(\tau)$ num.')
ax.plot(tau_lag, acv_teo, color=C_ANAL, lw=2.0, ls='--', label=r'$e^{-\beta\tau}$')
ax.axhline(0, color='gray', lw=0.8, ls=':')
ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'$C_v(\tau)$')
ax.set_title('Autocorrelación de velocidad'); ax.legend(fontsize=8)

# [2,2] PSD
ax = fig.add_subplot(gs[2,2])
ax.semilogy(freqs, psd_sm, color=C_EM, lw=1.8, label='PSD')
ax.axhline(S_teo_unit, color=C_RK4, ls='--', lw=1.6, label='Teórico')
ax.set_xlabel(r'$f$'); ax.set_ylabel('PSD')
ax.set_title(r'Espectro $\xi(t)$'); ax.legend(fontsize=8)

fig.suptitle(r'Movimiento Browniano — Panel Resumen'
             '\n' + r'$m=\gamma=k_BT=1$,  $v_0=2$,  $\Delta t=0.01$,  $N=2000$',
             fontsize=13)
fig.savefig('figuras/fig_panel_resumen.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_panel_resumen.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_panel_resumen")

# ---------------------------------------------------------------
# FIG 11: Error relativo posición (Euler vs RK4)
# ---------------------------------------------------------------
err_x_euler = np.abs(x_euler - x_anal)
err_x_rk4   = np.abs(x_rk4   - x_anal)
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.semilogy(t_array, err_x_euler+1e-20, color=C_EULER, lw=1.8,
            label=f'Euler  max={err_x_euler.max():.2e}')
ax.semilogy(t_array, err_x_rk4  +1e-20, color=C_RK4,   lw=1.8,
            label=f'RK4    max={err_x_rk4.max():.2e}')
ax.set_xlabel(r'$t$ [u.r.]'); ax.set_ylabel(r'$|x_n - x(t_n)|$')
ax.set_title(r'Error absoluto en posición — $\Delta t=0.01$')
ax.legend(); fig.tight_layout()
fig.savefig('figuras/fig_error_posicion.pdf', bbox_inches='tight')
fig.savefig('figuras/fig_error_posicion.png', bbox_inches='tight')
plt.close(fig)
print("  → fig_error_posicion")

# =============================================================================
# 12. RESUMEN ESTADÍSTICO FINAL
# =============================================================================
print("\n" + "="*62)
print("  RESUMEN FINAL")
print("="*62)
print(f"  {'Método':<12} {'Orden teórico':<18} {'Error máx. v(t)'}")
print(f"  {'Euler':<12} {'O(Δt¹)=1.00':<18} {err_euler_abs.max():.4e}")
print(f"  {'RK4':<12} {'O(Δt⁴)=4.00':<18} {err_rk4_abs.max():.4e}")
print(f"  {'EM':<12} {'O(√Δt)':<18} (estocástico)")
print()
print(f"  Pendiente Euler log-log : {sl_eu:.4f}  (teórica 1.0)")
print(f"  Pendiente RK4  log-log  : {sl_rk:.4f}  (teórica 4.0)")
print()
print(f"  D_teórico  = {D:.7f}")
print(f"  D_numérico = {D_num:.7f}")
print(f"  Error rel  = {abs(D_num-D)/D*100:.4f}%")
print(f"  R² (MSD)   = {r_m**2:.7f}")
print()
print(f"  KS gaussianidad t*={t_star}: stat={ks_stat:.4f}  p={ks_p:.4f}  "
      f"{'✓' if ks_p>0.05 else '✗'}")
print()
figs = sorted(os.listdir('figuras'))
print(f"  Figuras generadas ({len(figs)}):")
for f_ in figs:
    if f_.endswith('.png'):
        print(f"    {f_}")
print("="*62)
