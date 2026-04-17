"""
Fit a measured Pound-Drever-Hall error signal to extract the cavity
linewidth and the finesse.

Required inputs from the user (edit the CONFIG block below):
  * Excel file path with two columns: scan axis (time, piezo voltage,
    etc.) and error-signal voltage.
  * Modulation frequency Omega_m / 2 pi in Hz (what drives the EOM).
  * Free spectral range FSR in Hz (= c / 2 L for a two-mirror cavity).

The x-axis of the scan does NOT need to be calibrated in Hz; the two
PDH sidebands sit at +- Omega_m from the carrier, so the fit can
self-calibrate the scan axis from that known separation.

Output:
  * Fitted cavity linewidth kappa (FWHM, Hz)
  * Finesse = FSR / kappa (with 1-sigma uncertainty)
  * A diagnostic plot (data + fit + residuals)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from scipy.special import jv


# ===========================================================================
# CONFIG  --  edit these four lines for your measurement
# ===========================================================================
EXCEL_PATH = r'C:\Users\nahmad\Documents\Python Scripts\pdh_error.xlsx'
SHEET      = 0                # sheet index or name
X_COL      = 0                # column index (0-based) or header name of x
Y_COL      = 1                # column index (0-based) or header name of y

FSR_HZ     = 1.500e9          # cavity free spectral range in Hz  (edit me)
OMEGA_M_HZ = 20.0e6           # EOM modulation frequency in Hz    (edit me)

# optional hints (only affect initial guess, not final fit)
FINESSE_GUESS = 1000.0
BETA_GUESS    = 1.0
# ===========================================================================


# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------
def cavity_reflection(delta_hz, finesse, fsr_hz):
    r = np.exp(-np.pi / (2.0 * finesse))
    phi = 2.0 * np.pi * delta_hz / fsr_hz
    return (r * (np.exp(1j * phi) - 1.0)) / (1.0 - r**2 * np.exp(1j * phi))


def pdh_error_hz(delta_hz, finesse, fsr_hz, omega_m_hz, beta):
    F0 = cavity_reflection(delta_hz, finesse, fsr_hz)
    Fp = cavity_reflection(delta_hz + omega_m_hz, finesse, fsr_hz)
    Fm = cavity_reflection(delta_hz - omega_m_hz, finesse, fsr_hz)
    return 2.0 * jv(0, beta) * jv(1, beta) * np.imag(
        F0 * np.conj(Fp) - np.conj(F0) * Fm
    )


def model(x, x0, cal_hz_per_x, finesse, beta, scale, offset):
    """Full PDH error signal as a function of the raw scan axis x."""
    delta = cal_hz_per_x * (x - x0)
    return scale * pdh_error_hz(delta, finesse, FSR_HZ, OMEGA_M_HZ, beta) + offset


def run():
    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET)
    x_raw = df.iloc[:, X_COL].to_numpy() if isinstance(X_COL, int) else df[X_COL].to_numpy()
    y_raw = df.iloc[:, Y_COL].to_numpy() if isinstance(Y_COL, int) else df[Y_COL].to_numpy()

    order = np.argsort(x_raw)
    x = x_raw[order].astype(float)
    y = y_raw[order].astype(float)

    # ---------------------------------------------------------------------
    # Initial guesses from the data
    # ---------------------------------------------------------------------
    win = max(5, (len(y) // 200) | 1)
    y_s = savgol_filter(y, win, 3) if len(y) > win else y.copy()

    amp = 0.5 * (y_s.max() - y_s.min())
    offset0 = 0.5 * (y_s.max() + y_s.min())

    dist = max(1, len(y) // 50)
    pk_hi, _ = find_peaks(y_s - offset0, distance=dist, prominence=0.2 * amp)
    pk_lo, _ = find_peaks(-(y_s - offset0), distance=dist, prominence=0.2 * amp)
    all_pk = np.sort(np.concatenate([pk_hi, pk_lo]))

    if len(all_pk) >= 4:
        x_outer_lo = x[all_pk[0]]
        x_outer_hi = x[all_pk[-1]]
        cal_guess = (2.0 * OMEGA_M_HZ) / (x_outer_hi - x_outer_lo)
        x0_guess = 0.5 * (x_outer_lo + x_outer_hi)
    else:
        cal_guess = 4.0 * OMEGA_M_HZ / (x.max() - x.min())
        x0_guess = 0.5 * (x.max() + x.min())

    p0 = [x0_guess, cal_guess, FINESSE_GUESS, BETA_GUESS, amp, offset0]
    bounds_lo = [x.min(), -np.inf, 10.0,  0.1, 0.0,      y.min()]
    bounds_hi = [x.max(),  np.inf, 1.0e6, 5.0, 10 * amp, y.max()]

    popt, pcov = curve_fit(model, x, y, p0=p0,
                           bounds=(bounds_lo, bounds_hi), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    x0_f, cal_f, finesse_f, beta_f, scale_f, offset_f = popt
    x0_e, cal_e, finesse_e, beta_e, scale_e, offset_e = perr

    kappa_hz = FSR_HZ / finesse_f
    kappa_err = FSR_HZ / finesse_f**2 * finesse_e

    print('\n=== PDH fit results ===')
    print(f'  Carrier position  x0   = {x0_f:.6g}  +- {x0_e:.2g}')
    print(f'  Calibration            = {cal_f:.6g}  +- {cal_e:.2g} Hz / x-unit')
    print(f'  Modulation depth  beta = {beta_f:.3f}  +- {beta_e:.3f} rad')
    print(f'  Amplitude scale        = {scale_f:.3g}  offset = {offset_f:.3g}')
    print(f'  Cavity linewidth kappa = {kappa_hz:.4g} Hz  +- {kappa_err:.2g} Hz')
    print(f'  Finesse          F     = {finesse_f:.1f}  +- {finesse_e:.1f}')
    print(f'  (using FSR = {FSR_HZ:.4g} Hz, Omega_m/2pi = {OMEGA_M_HZ:.4g} Hz)')

    # ---------------------------------------------------------------------
    # Plot data + fit + residuals
    # ---------------------------------------------------------------------
    x_dense = np.linspace(x.min(), x.max(), 4000)
    y_fit_dense = model(x_dense, *popt)
    y_fit = model(x, *popt)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6),
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(x, y, '.', ms=3, color='0.5', label='data')
    ax1.plot(x_dense, y_fit_dense, 'C3', lw=1.5, label='PDH fit')
    ax1.axvline(x0_f, color='C0', lw=0.6, ls=':', label='carrier')
    ax1.axvline(x0_f + OMEGA_M_HZ / cal_f, color='C2', lw=0.6, ls=':',
                label=r'$\pm\Omega_m$')
    ax1.axvline(x0_f - OMEGA_M_HZ / cal_f, color='C2', lw=0.6, ls=':')
    ax1.set_ylabel('Error signal (V)')
    ax1.legend(loc='best', fontsize=8)
    ax1.set_title(f'Fit: $\\kappa$ = {kappa_hz:.3g} Hz, '
                  f'finesse = {finesse_f:.0f} $\\pm$ {finesse_e:.0f}')

    ax2.plot(x, y - y_fit, '.', ms=2, color='0.4')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('Scan axis (raw units)')
    ax2.set_ylabel('Residuals')

    out_png = os.path.join(os.path.dirname(os.path.abspath(EXCEL_PATH)),
                           'pdh_fit.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f'\nPlot saved to {out_png}')
    plt.show()
    return popt, perr


if __name__ == '__main__':
    run()
