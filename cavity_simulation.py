#!/usr/bin/env python3
"""
Gaussian-mode simulation of a planar-concave (half-symmetric) optical cavity.

Geometry
--------
    Mirror 1 : planar   (R1 = infinity)  -> the beam waist sits ON this mirror
    Mirror 2 : concave  (R2 = R)
    Cavity length L, default R = 1 m, L = 5 cm.

Formulas used (the "which formula" question)
--------------------------------------------
For a planar-concave cavity the g-parameters are

    g1 = 1 - L/R1 = 1          (planar mirror)
    g2 = 1 - L/R2 = 1 - L/R

and the cavity is stable for 0 <= g1*g2 <= 1, i.e. 0 < L < R.

Because the wavefront must be flat at the planar mirror, the waist is
located exactly at the planar mirror. The Rayleigh range is fixed purely
by the geometry:

    z_R = sqrt( L (R - L) )                                   [m]

and the wavelength enters only through

    w0     = sqrt( lambda * z_R / pi )                        waist (planar mirror)
    w(z)   = w0 * sqrt( 1 + (z/z_R)^2 )                       envelope inside cavity
    w_m2   = w0 * sqrt( 1 + (L/z_R)^2 )
           = sqrt( (lambda/pi) * sqrt( L R^2 / (R - L) ) )    spot on curved mirror
    theta  = lambda / (pi * w0)                               far-field half-angle

Longitudinal / transverse mode structure:

    FSR        = c / (2 L)
    delta_nu_T = (FSR/pi) * arccos( sqrt(g1*g2) )             transverse spacing
    nu(q,m,n)  = FSR * ( q + (m+n+1)/pi * arccos(sqrt(g1*g2)) )

Run:
    python3 cavity_simulation.py            # table + plots (saved as PNG)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 299_792_458.0  # speed of light [m/s]


class PlanarConcaveCavity:
    """Planar-concave two-mirror cavity: flat mirror at z=0, concave (radius R) at z=L."""

    def __init__(self, R=1.0, L=0.05):
        if not 0 < L < R:
            raise ValueError(f"Unstable/degenerate geometry: need 0 < L < R (got L={L}, R={R})")
        self.R = R
        self.L = L
        self.g1 = 1.0
        self.g2 = 1.0 - L / R
        self.z_R = np.sqrt(L * (R - L))  # Rayleigh range, waist at flat mirror

    # ---- wavelength-dependent quantities (lam in metres; accepts arrays) ----
    def waist(self, lam):
        """Beam waist w0 on the planar mirror [m]."""
        return np.sqrt(np.asarray(lam) * self.z_R / np.pi)

    def spot(self, lam, z):
        """Beam radius w(z) at position z from the planar mirror [m]."""
        return self.waist(lam) * np.sqrt(1.0 + (np.asarray(z) / self.z_R) ** 2)

    def spot_on_curved_mirror(self, lam):
        return self.spot(lam, self.L)

    def divergence(self, lam):
        """Far-field half-angle divergence [rad]."""
        return np.asarray(lam) / (np.pi * self.waist(lam))

    def mode_volume(self, lam):
        """Approximate TEM00 mode volume V = (pi/4) w0^2 L [m^3]."""
        return np.pi / 4.0 * self.waist(lam) ** 2 * self.L

    # ---- geometry-only quantities ----
    @property
    def fsr(self):
        """Free spectral range [Hz]."""
        return C / (2.0 * self.L)

    @property
    def gouy(self):
        """Round-trip single-pass Gouy phase arccos(sqrt(g1 g2)) [rad]."""
        return np.arccos(np.sqrt(self.g1 * self.g2))

    @property
    def transverse_mode_spacing(self):
        """Frequency spacing between TEM(m+n) and TEM(m+n+1) [Hz]."""
        return self.fsr * self.gouy / np.pi

    def resonance(self, q, m=0, n=0):
        """Resonance frequency of the TEM_mn mode with longitudinal index q [Hz]."""
        return self.fsr * (q + (m + n + 1) * self.gouy / np.pi)


def print_table(cav, wavelengths_nm):
    print(f"Planar-concave cavity: R = {cav.R:.3f} m, L = {cav.L * 100:.1f} cm")
    print(f"  g1*g2 = {cav.g1 * cav.g2:.3f}  (stable, 0 <= g1*g2 <= 1)")
    print(f"  Rayleigh range z_R = sqrt(L(R-L)) = {cav.z_R * 100:.2f} cm")
    print(f"  FSR = c/2L = {cav.fsr / 1e9:.3f} GHz")
    print(f"  Transverse mode spacing = {cav.transverse_mode_spacing / 1e6:.1f} MHz")
    print()
    hdr = f"{'lambda [nm]':>12} {'w0 flat [um]':>13} {'w curved [um]':>14} {'theta [mrad]':>13} {'V_mode [mm^3]':>14}"
    print(hdr)
    print("-" * len(hdr))
    for lam_nm in wavelengths_nm:
        lam = lam_nm * 1e-9
        print(
            f"{lam_nm:>12.0f} {cav.waist(lam) * 1e6:>13.1f} "
            f"{cav.spot_on_curved_mirror(lam) * 1e6:>14.1f} "
            f"{cav.divergence(lam) * 1e3:>13.3f} "
            f"{cav.mode_volume(lam) * 1e9:>14.4f}"
        )


def _colors(wavelengths_nm):
    return plt.cm.viridis(np.linspace(0.0, 0.85, len(wavelengths_nm)))


def plot_envelope(ax, cav, wavelengths_nm):
    """(1) Beam envelope inside the cavity for the selected wavelengths."""
    colors = _colors(wavelengths_nm)
    z = np.linspace(0, cav.L, 400)
    for lam_nm, c in zip(wavelengths_nm, colors):
        w = cav.spot(lam_nm * 1e-9, z) * 1e6
        ax.plot(z * 100, w, color=c, label=f"{lam_nm:.0f} nm")
        ax.plot(z * 100, -w, color=c)
    ax.axvline(0, color="k", lw=2)
    ax.axvline(cav.L * 100, color="k", lw=2, ls="--")
    ax.set_xlabel("z from planar mirror [cm]")
    ax.set_ylabel(r"beam radius $\pm w(z)$ [$\mu$m]")
    ax.set_title("TEM$_{00}$ mode envelope (waist on flat mirror)")
    ax.legend(fontsize=8, title="wavelength")
    ax.grid(alpha=0.3)


def plot_spots(ax, cav, wavelengths_nm):
    """(2) Waist and mirror spot vs wavelength."""
    colors = _colors(wavelengths_nm)
    lam_scan = np.linspace(300, 1700, 500) * 1e-9
    ax.plot(lam_scan * 1e9, cav.waist(lam_scan) * 1e6, label=r"$w_0$ (planar mirror)")
    ax.plot(lam_scan * 1e9, cav.spot_on_curved_mirror(lam_scan) * 1e6, label=r"$w$ (curved mirror)")
    for lam_nm, c in zip(wavelengths_nm, colors):
        ax.plot(lam_nm, cav.waist(lam_nm * 1e-9) * 1e6, "o", color=c)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel(r"beam radius [$\mu$m]")
    ax.set_title(r"Spot sizes $\propto \sqrt{\lambda}$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_divergence(ax, cav, wavelengths_nm):
    """(3) Divergence vs wavelength."""
    colors = _colors(wavelengths_nm)
    lam_scan = np.linspace(300, 1700, 500) * 1e-9
    ax.plot(lam_scan * 1e9, cav.divergence(lam_scan) * 1e3, color="tab:red")
    for lam_nm, c in zip(wavelengths_nm, colors):
        ax.plot(lam_nm, cav.divergence(lam_nm * 1e-9) * 1e3, "o", color=c)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("far-field half angle [mrad]")
    ax.set_title(r"Divergence $\theta = \lambda/\pi w_0 \propto \sqrt{\lambda}$")
    ax.grid(alpha=0.3)


def plot_spectrum(ax, cav, wavelengths_nm):
    """(4) Mode spectrum over one FSR (wavelength-independent)."""
    fsr = cav.fsr
    dt = cav.transverse_mode_spacing
    for order, (h, c) in enumerate(zip([1.0, 0.6, 0.35], ["k", "tab:blue", "tab:orange"])):
        # offsets relative to the TEM00 line, shown for two adjacent longitudinal orders q
        positions = [q * fsr + order * dt for q in (0, 1)]
        positions = [p for p in positions if p <= 1.05 * fsr]
        ax.vlines(np.array(positions) / 1e9, 0, h, color=c, label=f"TEM, m+n={order}")
    ax.set_xlim(-0.05 * fsr / 1e9, 1.05 * fsr / 1e9)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("frequency offset within one FSR [GHz]")
    ax.set_ylabel("relative amplitude (schematic)")
    ax.set_title(
        f"Mode spectrum: FSR = {fsr / 1e9:.2f} GHz, "
        f"$\\Delta\\nu_T$ = {dt / 1e6:.0f} MHz"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


# name -> plotting function; each draws one panel onto a given axis
PANELS = {
    "envelope": plot_envelope,
    "spots": plot_spots,
    "divergence": plot_divergence,
    "spectrum": plot_spectrum,
}


def _suptitle(cav):
    return (
        f"Planar-concave cavity  (R = {cav.R:.0f} m, L = {cav.L * 100:.0f} cm, "
        f"$z_R$ = {cav.z_R * 100:.1f} cm)"
    )


def make_combined_figure(cav, wavelengths_nm, fname="cavity_simulation.png"):
    """All four panels in one 2x2 figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(_suptitle(cav), fontsize=13)
    for ax, plot_fn in zip(axes.flat, PANELS.values()):
        plot_fn(ax, cav, wavelengths_nm)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def make_single_figure(name, cav, wavelengths_nm, fname=None):
    """One panel as its own figure, saved to <name>.png by default."""
    fname = fname or f"cavity_{name}.png"
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.suptitle(_suptitle(cav), fontsize=11)
    PANELS[name](ax, cav, wavelengths_nm)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Planar-concave cavity Gaussian mode simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-R", "--radius", type=float, default=1.0,
                        help="radius of curvature of the concave mirror [m]")
    parser.add_argument("-L", "--length", type=float, default=0.05,
                        help="cavity length [m]")
    parser.add_argument("-w", "--wavelengths", type=float, nargs="+",
                        default=[405, 532, 633, 780, 1064, 1550],
                        help="wavelengths to highlight [nm]")
    parser.add_argument("-p", "--plots", nargs="+",
                        choices=[*PANELS, "all"], default=["all"],
                        help="which plots to generate")
    parser.add_argument("--separate", action="store_true",
                        help="save each selected plot as its own PNG "
                             "(automatic when specific plots are chosen)")
    parser.add_argument("--no-table", action="store_true",
                        help="skip the printed results table")
    args = parser.parse_args()

    cav = PlanarConcaveCavity(R=args.radius, L=args.length)
    if not args.no_table:
        print_table(cav, args.wavelengths)
        print()

    selected = list(PANELS) if "all" in args.plots else args.plots
    if "all" in args.plots and not args.separate:
        make_combined_figure(cav, args.wavelengths)
    else:
        for name in selected:
            make_single_figure(name, cav, args.wavelengths)


if __name__ == "__main__":
    main()
