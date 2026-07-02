"""Tests for the overlapping Allan-deviation implementation.

The module's estimator is cross-checked against an independent,
brute-force implementation of the standard frequency-domain overlapping
Allan-variance formula, and against known analytical behaviour (white FM
noise has an ADEV slope of -1/2 on a log-log tau axis).
"""

import numpy as np
import pytest

from fxe_analyzer import stability


def brute_force_oavar(y, m):
    """Reference overlapping Allan variance from the frequency formula.

    sigma^2 = 1/(2 m^2 (N-2m+1)) * sum_j [ sum_{i=j}^{j+m-1} (y_{i+m}-y_i) ]^2
    """
    y = np.asarray(y, dtype=float)
    N = y.size
    if N < 2 * m + 1:
        return float("nan")
    total = 0.0
    for j in range(0, N - 2 * m + 1):
        inner = 0.0
        for i in range(j, j + m):
            inner += y[i + m] - y[i]
        total += inner * inner
    return total / (2.0 * m * m * (N - 2 * m + 1))


@pytest.mark.parametrize("m", [1, 2, 3, 5, 10])
def test_matches_brute_force(m):
    rng = np.random.default_rng(42)
    y = rng.standard_normal(500) + 1000.0  # offset to check invariance too
    avar, n = stability.overlapping_avar(y, m)
    ref = brute_force_oavar(y, m)
    assert n == y.size - 2 * m + 1
    assert avar == pytest.approx(ref, rel=1e-9)


def test_offset_invariance():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(300)
    a1, _ = stability.overlapping_adev(y, 4)
    a2, _ = stability.overlapping_adev(y + 12345.6789, 4)
    assert a1 == pytest.approx(a2, rel=1e-9)


def test_m1_equals_simple_definition():
    # At m=1 the overlapping Allan variance reduces to the classic
    # 1/(2(N-1)) * sum (y_{i+1}-y_i)^2.
    rng = np.random.default_rng(7)
    y = rng.standard_normal(200)
    avar, _ = stability.overlapping_avar(y, 1)
    d = np.diff(y)
    simple = np.sum(d * d) / (2.0 * (y.size - 1))
    assert avar == pytest.approx(simple, rel=1e-12)


def test_white_fm_slope_is_minus_half():
    # White frequency noise -> ADEV ∝ tau^(-1/2).
    rng = np.random.default_rng(123)
    n = 200_000
    tau0 = 1e-3
    y = rng.standard_normal(n)  # white FM
    pts = stability.compute_stability(
        y, tau0, taus=[1e-3, 10e-3, 100e-3, 1.0], f_ref=1.0
    )
    taus = np.array([p.tau_used for p in pts])
    adev = np.array([p.adev_hz for p in pts])
    slope = np.polyfit(np.log10(taus), np.log10(adev), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.05)


def test_fractional_scaling():
    rng = np.random.default_rng(5)
    y = rng.standard_normal(1000) + 5.0e6
    f_ref = 5.0e6
    pts = stability.compute_stability(y, 1e-3, taus=[1e-3, 10e-3], f_ref=f_ref)
    for p in pts:
        assert p.adev_frac == pytest.approx(p.adev_hz / f_ref, rel=1e-12)


def test_skips_taus_without_enough_data():
    y = np.random.default_rng(1).standard_normal(50)
    # 10 s at 1 ms gate needs >20000 samples; must be skipped.
    pts = stability.compute_stability(y, 1e-3, taus=[1e-3, 10.0], f_ref=1.0)
    used = {round(p.tau, 6) for p in pts}
    assert 1e-3 in used
    assert 10.0 not in used


def test_tau_shorter_than_gate_skipped():
    y = np.random.default_rng(2).standard_normal(1000)
    pts = stability.compute_stability(y, 1e-2, taus=[1e-3], f_ref=1.0)
    assert pts == []  # tau < tau0 -> m rounds to 0 -> skipped


def test_default_taus_present():
    assert stability.DEFAULT_TAUS == (1e-3, 10e-3, 100e-3, 200e-3, 1.0, 10.0)


def test_detrend_removes_linear_drift():
    tau0 = 1e-3
    n = 10_000
    t = np.arange(n) * tau0
    drift = 3.0  # Hz/s
    rng = np.random.default_rng(1)
    y = 5.0e6 + drift * t + 0.5 * rng.standard_normal(n)
    res, coeffs = stability.detrend_series(y, tau0, 1)
    assert coeffs[0] == pytest.approx(drift, rel=1e-2)     # recovered slope
    # residual is essentially flat
    assert stability.estimate_drift(res, tau0) == pytest.approx(0.0, abs=1e-2)


def test_estimate_drift():
    tau0 = 1e-3
    t = np.arange(5000) * tau0
    y = 1.0e6 - 2.5 * t
    assert stability.estimate_drift(y, tau0) == pytest.approx(-2.5, rel=1e-6)


def test_detrend_lowers_long_tau_adev():
    # A strong linear drift inflates the long-tau ADEV; removing it helps.
    tau0 = 1e-3
    n = 100_000
    t = np.arange(n) * tau0
    rng = np.random.default_rng(2)
    y = 1.0e6 + 50.0 * t + rng.standard_normal(n)  # big drift
    with_drift = stability.compute_stability(y, tau0, taus=[1.0], f_ref=1.0)[0]
    without = stability.compute_stability(y, tau0, taus=[1.0], f_ref=1.0, detrend=1)[0]
    assert without.adev_hz < with_drift.adev_hz * 0.5


def test_detrend_fref_from_original_mean():
    tau0 = 1e-3
    t = np.arange(5000) * tau0
    y = 4.0e6 + 10.0 * t + np.random.default_rng(3).standard_normal(5000)
    mean0 = float(np.mean(y))
    pts = stability.compute_stability(y, tau0, taus=[10e-3], f_ref=None, detrend=1)
    p = pts[0]
    assert p.adev_frac == pytest.approx(p.adev_hz / mean0, rel=1e-9)


def test_allan_slope_and_classification():
    # White FM -> slope ~ -0.5 -> "White FM".
    rng = np.random.default_rng(9)
    y = rng.standard_normal(200_000)
    pts = stability.compute_stability(y, 1e-3, taus=[1e-3, 10e-3, 100e-3, 1.0], f_ref=1.0)
    slope = stability.allan_slope([p.tau_used for p in pts], [p.adev_hz for p in pts])
    assert slope == pytest.approx(-0.5, abs=0.05)
    assert stability.classify_noise(slope) == "White FM"
    assert stability.classify_noise(1.0) == "Frequency drift"
    assert stability.classify_noise(0.0) == "Flicker FM"
    assert stability.classify_noise(float("nan")) == "—"
