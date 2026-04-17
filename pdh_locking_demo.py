"""
Pound-Drever-Hall (PDH) laser frequency locking: a teaching simulation.

Signal chain rendered as six linked panels:
  1. Input laser spectrum after phase modulation (carrier + sidebands)
  2. Fabry-Perot cavity reflection coefficient (|F| and arg F)
  3. Time-domain RF photodiode signal at/near resonance
  4. Demodulation: photodiode * local oscillator, before/after low-pass filter
  5. Classic PDH error signal vs laser detuning (sweep)
  6. Closed-loop behaviour: a disturbance rejected by an integral servo

Units: cavity linewidth kappa = 1, FSR = finesse * kappa, time in 1/kappa.
Numbers are chosen for pedagogical clarity, not any specific apparatus.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.special import jv

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'pdh_locking_demo.png')


# ---------------------------------------------------------------------------
# Physical model (all frequencies in units of the cavity linewidth kappa)
# ---------------------------------------------------------------------------
KAPPA = 1.0
FINESSE = 300.0
FSR = FINESSE * KAPPA                           # free spectral range
OMEGA_M = 6.0 * KAPPA                           # modulation frequency
BETA = 1.08                                     # modulation depth (J0*J1 peak)


def cavity_reflection(delta):
    """Reflection coefficient F(Delta) of a symmetric Fabry-Perot cavity."""
    r = np.exp(-np.pi / (2.0 * FINESSE))
    phi = 2.0 * np.pi * delta / FSR
    return (r * (np.exp(1j * phi) - 1.0)) / (1.0 - r**2 * np.exp(1j * phi))


def pdh_error(delta):
    """Analytic PDH in-phase error signal (small-modulation, three-tone)."""
    F0 = cavity_reflection(delta)
    Fp = cavity_reflection(delta + OMEGA_M)
    Fm = cavity_reflection(delta - OMEGA_M)
    return 2.0 * jv(0, BETA) * jv(1, BETA) * np.imag(
        F0 * np.conj(Fp) - np.conj(F0) * Fm
    )


def reflected_power(delta, t):
    """Instantaneous reflected optical power for quasi-static detuning delta."""
    F0 = cavity_reflection(delta)
    Fp = cavity_reflection(delta + OMEGA_M)
    Fm = cavity_reflection(delta - OMEGA_M)
    E0 = jv(0, BETA) * F0
    Ep = jv(1, BETA) * Fp
    Em = -jv(1, BETA) * Fm
    phi = OMEGA_M * t
    E = E0 + Ep * np.exp(1j * phi) + Em * np.exp(-1j * phi)
    return np.abs(E) ** 2


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.28)

# --- Panel 1: modulated laser spectrum -------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
for n in range(-3, 4):
    amp = jv(n, BETA)
    color = 'C0' if n == 0 else ('C3' if abs(n) == 1 else 'C7')
    ax1.vlines(n * OMEGA_M, 0, amp**2, color=color, lw=3)
ax1.axhline(0, color='k', lw=0.5)
ax1.set_xlabel(r'Frequency offset $(\omega-\omega_0)/\kappa$')
ax1.set_ylabel('Power (a.u.)')
ax1.set_title('1. Phase-modulated laser: carrier (blue) + sidebands (red)')
ax1.set_xlim(-3.5 * OMEGA_M, 3.5 * OMEGA_M)

# --- Panel 2: cavity reflection --------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
d = np.linspace(-10, 10, 4000)
F = cavity_reflection(d)
ax2.plot(d, np.abs(F), 'C0', label=r'$|F(\omega)|$')
ax2.set_xlabel(r'Detuning $\Delta/\kappa$')
ax2.set_ylabel(r'$|F|$', color='C0')
ax2.tick_params(axis='y', labelcolor='C0')
ax2b = ax2.twinx()
ax2b.plot(d, np.angle(F), 'C3', label=r'$\arg F$')
ax2b.set_ylabel(r'$\arg F$ (rad)', color='C3')
ax2b.tick_params(axis='y', labelcolor='C3')
ax2.set_title('2. Cavity reflection: amplitude dip, phase flip')

# --- Panel 3: time-domain photodiode RF signal -----------------------------
ax3 = fig.add_subplot(gs[1, 0])
# Sample fast enough to resolve OMEGA_M; show ~6 RF cycles
T_rf = 6 * (2 * np.pi / OMEGA_M)
fs = 50 * OMEGA_M / (2 * np.pi)                 # samples per kappa^-1
t = np.arange(0, T_rf, 1.0 / fs)

detunings = {r'on resonance, $\Delta=0$': 0.0,
             r'red-detuned, $\Delta=-0.4\kappa$': -0.4,
             r'blue-detuned, $\Delta=+0.4\kappa$': +0.4}
colors = {r'on resonance, $\Delta=0$': 'C0',
          r'red-detuned, $\Delta=-0.4\kappa$': 'C3',
          r'blue-detuned, $\Delta=+0.4\kappa$': 'C2'}

for label, dv in detunings.items():
    p = reflected_power(dv, t)
    ax3.plot(t, p, colors[label], lw=1.1, label=label)
ax3.set_xlabel(r'Time $\kappa t$')
ax3.set_ylabel('Photodiode signal (a.u.)')
ax3.set_title(f'3. RF photodiode signal at $\\Omega_m={OMEGA_M:.0f}\\kappa$')
ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)

# --- Panel 4: demodulation + low-pass --------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
T_long = 20 * (2 * np.pi / OMEGA_M)
t2 = np.arange(0, T_long, 1.0 / fs)
sos = butter(4, OMEGA_M / (2 * np.pi) / 4.0, 'low', fs=fs, output='sos')
lo = np.sin(OMEGA_M * t2)                       # quadrature LO -> dispersive err
for label, dv in detunings.items():
    p = reflected_power(dv, t2)
    mixed = (p - p.mean()) * lo
    err = sosfiltfilt(sos, mixed)
    ax4.plot(t2, err, colors[label], lw=1.4, label=label)
ax4.axhline(0, color='k', lw=0.5)
ax4.set_xlabel(r'Time $\kappa t$')
ax4.set_ylabel('Error signal (a.u.)')
ax4.set_title('4. Mixer output after low-pass filter (sign = side of resonance)')
ax4.legend(loc='center right', fontsize=8, framealpha=0.9)

# --- Panel 5: analytic PDH error signal ------------------------------------
ax5 = fig.add_subplot(gs[2, 0])
sweep = np.linspace(-2.5 * OMEGA_M, 2.5 * OMEGA_M, 4000)
err = pdh_error(sweep)
ax5.plot(sweep, err, 'C0', lw=1.5)
ax5.axhline(0, color='k', lw=0.5)
ax5.axvline(0, color='k', lw=0.5, ls=':')
for s in (-1, 1):
    ax5.axvline(s * OMEGA_M, color='C3', lw=0.5, ls=':')
ax5.annotate('carrier\nresonance', xy=(0, 0), xytext=(1.3, 0.45),
             textcoords='data', fontsize=8, color='C0',
             arrowprops=dict(arrowstyle='->', color='C0', lw=0.7))
ax5.annotate(r'sideband at $-\Omega_m$', xy=(-OMEGA_M, 0),
             xytext=(-2.2 * OMEGA_M, -0.45), fontsize=8, color='C3',
             arrowprops=dict(arrowstyle='->', color='C3', lw=0.7))
ax5.set_xlabel(r'Laser detuning $\Delta/\kappa$')
ax5.set_ylabel(r'$\varepsilon(\Delta)$')
ax5.set_title('5. PDH error signal: steep linear slope through the carrier')

# --- Panel 6: closed-loop disturbance rejection ----------------------------
ax6 = fig.add_subplot(gs[2, 1])
dt = 0.02                                       # in units of 1/kappa
t6 = np.arange(0, 200.0, dt)

# Free-running laser frequency noise: slow drift + acoustic tone
drift = 0.8 * np.sin(2 * np.pi * t6 / 120.0)
acoustic = 0.25 * np.sin(2 * np.pi * t6 / 8.0)
nu_free = drift + acoustic

# Integral servo on the PDH error signal. Sign matches the (negative)
# error-signal slope at the carrier so that feedback is corrective.
ki = 3.0
nu_locked = np.zeros_like(t6)
servo = 0.0
for i, n_free in enumerate(nu_free):
    nu = n_free - servo
    e = pdh_error(np.array([nu]))[0]
    servo -= ki * e * dt
    nu_locked[i] = nu

ax6.plot(t6, nu_free, 'C7', lw=1.0, label='free-running laser')
ax6.plot(t6, nu_locked, 'C0', lw=1.3, label='locked laser')
ax6.axhline(0, color='k', lw=0.5)
ax6.set_xlabel(r'Time $\kappa t$')
ax6.set_ylabel(r'Detuning $\Delta/\kappa$')
ax6.set_title('6. Servo closed: drift + acoustic tone suppressed')
ax6.legend(loc='upper right', fontsize=8, framealpha=0.9)

fig.suptitle('Pound-Drever-Hall laser frequency locking: signal chain',
             fontsize=14, y=0.995)
fig.savefig(OUT_PATH, dpi=140, bbox_inches='tight')
print(f'Saved {OUT_PATH}')
plt.show()
