"""Frequency-stability calculations.

The key quantity for a frequency counter / beat-note measurement is the
*fractional frequency stability*, characterised by the Allan deviation
(ADEV).  This module computes the **overlapping** Allan deviation, which
is the standard best estimator, and returns both the absolute stability
(in Hz) and the fractional stability (dimensionless) at any requested
averaging time tau.

Definitions
-----------
Given N frequency samples ``f_i`` taken with a uniform gate time
``tau0`` (seconds), and an averaging factor ``m`` (so that
``tau = m * tau0``), the overlapping Allan variance is

    sigma^2(tau) = 1 / (2 m^2 (N - 2m + 1))
                   * sum_{j=1}^{N-2m+1} [ sum_{i=j}^{j+m-1} (f_{i+m} - f_i) ]^2

The Allan deviation is ``sqrt`` of that.  Because the estimator is built
from *differences* of the samples it is invariant to any constant offset
(so subtracting the mean, or not, gives the same result).  This lets us
compute the ADEV of the raw frequency (in Hz) and obtain the fractional
stability simply by dividing by a reference frequency ``f_ref``:

    adev_fractional(tau) = adev_hz(tau) / f_ref

``f_ref`` is the carrier the beat is referenced to (e.g. the optical
frequency).  If it is unknown, using the mean of the channel gives the
stability of the beat relative to itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

# Convenient default averaging times (seconds) requested for FXE beats.
DEFAULT_TAUS: tuple = (1e-3, 10e-3, 100e-3, 200e-3, 1.0, 10.0)

# Power-law noise types by ADEV log-log slope  (sigma_y(tau) ~ tau^slope).
# ADEV cannot separate White PM from Flicker PM (both ~tau^-1); MDEV can.
NOISE_TABLE = [
    (-1.0, "White/Flicker PM"),
    (-0.5, "White FM"),
    (0.0, "Flicker FM"),
    (0.5, "Random-walk FM"),
    (1.0, "Frequency drift"),
]


@dataclass
class StabilityPoint:
    """Stability result at a single averaging time."""

    tau: float          # requested averaging time (s)
    m: int              # averaging factor actually used (tau_used = m * tau0)
    tau_used: float     # m * tau0  (may differ slightly from `tau`)
    adev_hz: float      # absolute Allan deviation (Hz)
    adev_frac: float    # fractional Allan deviation (adev_hz / f_ref)
    err_hz: float       # approximate 1-sigma uncertainty on adev_hz
    err_frac: float     # approximate 1-sigma uncertainty on adev_frac
    n: int              # number of analysis terms (pairs) used


def frequency_to_phase(freq: np.ndarray, tau0: float) -> np.ndarray:
    """Integrate frequency samples into phase (time error) samples.

    Returns an array of length ``len(freq) + 1`` starting at 0.
    """
    freq = np.asarray(freq, dtype=float)
    phase = np.empty(freq.size + 1, dtype=float)
    phase[0] = 0.0
    np.cumsum(freq * tau0, out=phase[1:])
    return phase


def overlapping_avar(freq: np.ndarray, m: int) -> tuple[float, int]:
    """Overlapping Allan *variance* of ``freq`` at averaging factor ``m``.

    The result is expressed in the (squared) units of ``freq``.  It is
    independent of the gate time because ``freq`` is treated as a series
    of samples; ``m`` alone sets the averaging.

    Returns ``(avar, n_terms)``.  ``avar`` is ``nan`` when there is not
    enough data (need ``len(freq) >= 2*m``).
    """
    freq = np.asarray(freq, dtype=float)
    n = freq.size
    if m < 1 or n < 2 * m + 1:
        return float("nan"), 0

    # Phase-difference (second difference) formulation is numerically the
    # cleanest and equals the frequency formula above.
    phase = frequency_to_phase(freq, 1.0)  # tau0 folded out; cancels below
    # second difference: x[i+2m] - 2 x[i+m] + x[i]
    d = phase[2 * m:] - 2.0 * phase[m:-m] + phase[:-2 * m]
    n_terms = d.size  # = (n + 1) - 2m
    avar = np.sum(d * d) / (2.0 * (m ** 2) * n_terms)
    return float(avar), int(n_terms)


def overlapping_adev(freq: np.ndarray, m: int) -> tuple[float, int]:
    """Overlapping Allan deviation (sqrt of variance)."""
    avar, n = overlapping_avar(freq, m)
    return (float(np.sqrt(avar)) if np.isfinite(avar) else float("nan")), n


def _m_for_tau(tau: float, tau0: float) -> int:
    """Averaging factor for a requested tau given gate time tau0."""
    return int(round(tau / tau0))


def detrend_series(freq, tau0: float, order: int):
    """Remove a polynomial trend of ``order`` from ``freq`` (1=linear drift).

    The fit is done against a uniform time axis (t = i·tau0) using only the
    finite samples; NaNs are preserved in the residual. Returns
    ``(residual, coeffs)`` where ``coeffs`` are the polynomial coefficients
    in time (highest power first), or ``(freq, None)`` when ``order < 1`` or
    there are too few points.
    """
    freq = np.asarray(freq, dtype=float)
    if order < 1:
        return freq, None
    n = freq.size
    t = np.arange(n) * tau0
    finite = np.isfinite(freq)
    if int(finite.sum()) <= order:
        return freq, None
    coeffs = np.polyfit(t[finite], freq[finite], order)
    return freq - np.polyval(coeffs, t), coeffs


def estimate_drift(freq, tau0: float) -> float:
    """Linear frequency drift rate in Hz per second (NaN if indeterminate)."""
    _, coeffs = detrend_series(freq, tau0, 1)
    return float(coeffs[0]) if coeffs is not None else float("nan")


def allan_slope(taus, adevs) -> float:
    """Log-log slope of sigma(tau); NaN if fewer than two valid points."""
    taus = np.asarray(taus, dtype=float)
    adevs = np.asarray(adevs, dtype=float)
    good = np.isfinite(taus) & np.isfinite(adevs) & (taus > 0) & (adevs > 0)
    if int(good.sum()) < 2:
        return float("nan")
    return float(np.polyfit(np.log10(taus[good]), np.log10(adevs[good]), 1)[0])


def classify_noise(slope: float) -> str:
    """Nearest power-law noise type for an ADEV log-log ``slope``."""
    if not np.isfinite(slope):
        return "—"
    return min(NOISE_TABLE, key=lambda kv: abs(kv[0] - slope))[1]


def compute_stability(
    freq: Sequence[float],
    tau0: float,
    taus: Iterable[float] = DEFAULT_TAUS,
    f_ref: Optional[float] = None,
    detrend: int = 0,
) -> List[StabilityPoint]:
    """Compute overlapping ADEV at each requested averaging time.

    Parameters
    ----------
    freq : sequence of float
        Frequency samples (Hz), uniformly spaced by ``tau0``.
    tau0 : float
        Gate time / sample interval (s).
    taus : iterable of float
        Averaging times to evaluate (s).
    f_ref : float, optional
        Reference/carrier frequency for the fractional stability.  If
        ``None`` the mean of ``freq`` is used (stability of the beat
        relative to itself).
    detrend : int
        Polynomial order of frequency trend to remove before the ADEV
        (0 = none, 1 = linear drift, 2 = quadratic).  ``f_ref`` for the
        fractional value is taken from the *original* mean.

    Returns
    -------
    list of StabilityPoint
        One entry per requested tau that has enough data (taus that are
        too long for the record, or shorter than ``tau0``, are skipped).
    """
    freq = np.asarray(list(freq), dtype=float)
    finite = freq[np.isfinite(freq)]
    if finite.size == 0 or tau0 <= 0:
        return []

    if f_ref is None:
        f_ref = float(np.mean(finite))  # from the original data, pre-detrend
    # Guard against a zero reference for the fractional value.
    safe_ref = f_ref if f_ref not in (0.0, None) else float("nan")

    if detrend and detrend >= 1:
        freq, _ = detrend_series(freq, tau0, detrend)
    freq = freq[np.isfinite(freq)]
    if freq.size == 0:
        return []

    results: List[StabilityPoint] = []
    for tau in sorted(set(taus)):
        m = _m_for_tau(tau, tau0)
        if m < 1:
            continue  # tau shorter than the gate time
        adev, n = overlapping_adev(freq, m)
        if not np.isfinite(adev) or n < 1:
            continue  # not enough data for this tau
        # Simple, clearly-approximate 1-sigma error estimate based on the
        # number of independent (non-overlapping) samples available.
        n_indep = max(1, freq.size // m - 1)
        err_hz = adev / np.sqrt(n_indep)
        results.append(
            StabilityPoint(
                tau=float(tau),
                m=int(m),
                tau_used=float(m * tau0),
                adev_hz=float(adev),
                adev_frac=float(adev / safe_ref),
                err_hz=float(err_hz),
                err_frac=float(err_hz / safe_ref),
                n=int(n),
            )
        )
    return results


def stability_table(
    channels: dict,
    tau0: float,
    taus: Iterable[float] = DEFAULT_TAUS,
    f_refs: Optional[dict] = None,
    detrend: int = 0,
) -> dict:
    """Compute stability for several channels at once.

    Parameters
    ----------
    channels : dict[str, sequence]
        Mapping of channel name -> frequency samples.
    tau0 : float
        Gate time (s), common to all channels.
    taus : iterable of float
        Averaging times (s).
    f_refs : dict[str, float], optional
        Per-channel reference frequency; falls back to the channel mean.

    Returns
    -------
    dict[str, list[StabilityPoint]]
    """
    f_refs = f_refs or {}
    out = {}
    for name, freq in channels.items():
        out[name] = compute_stability(
            freq, tau0, taus, f_ref=f_refs.get(name), detrend=detrend
        )
    return out
