#!/usr/bin/env python3
"""Generate synthetic FXE-style frequency-counter files for testing.

Produces plain-text tables with a Modified-Julian-Date timestamp column
followed by 8 frequency channels sampled at a 1 ms gate time.  Each
channel is a beat note near a few MHz with a mix of white and flicker
frequency noise, so the resulting Allan-deviation curves look realistic.

By default it writes two consecutive files (``fxe_beat_part1.txt`` and
``fxe_beat_part2.txt``) so the "join multiple files" path can be tested.

Usage::

    python sample_data/generate_sample.py            # defaults
    python sample_data/generate_sample.py --seconds 30 --outdir /tmp
"""

from __future__ import annotations

import argparse
import os

import numpy as np

MJD_UNIX_EPOCH = 40587.0
SECONDS_PER_DAY = 86400.0


def _flicker_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate flicker (1/f) noise by filtering white noise in freq."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    spectrum /= np.sqrt(f)  # 1/f power -> 1/sqrt(f) amplitude
    out = np.fft.irfft(spectrum, n=n)
    return out / (np.std(out) or 1.0)


def make_channels(
    n: int, gate: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Build 8 realistic beat-frequency channels (Hz)."""
    channels = []
    for ch in range(8):
        f0 = 1.0e6 + ch * 5.0e5              # nominal beat, 1.0–4.5 MHz
        white_amp = 20.0 + 10.0 * ch         # Hz, white FM level
        flicker_amp = 5.0 + 2.0 * ch         # Hz, flicker floor
        drift = (ch - 4) * 2.0               # Hz/s linear drift
        t = np.arange(n) * gate
        series = (
            f0
            + white_amp * rng.standard_normal(n)
            + flicker_amp * _flicker_noise(n, rng)
            + drift * t
        )
        channels.append(series)
    return channels


def write_file(
    path: str,
    start_mjd: float,
    n: int,
    gate: float,
    rng: np.random.Generator,
) -> None:
    channels = make_channels(n, gate, rng)
    mjd = start_mjd + (np.arange(n) * gate) / SECONDS_PER_DAY
    matrix = np.column_stack([mjd] + channels)
    header = (
        "# FXE frequency counter export (synthetic)\n"
        f"# gate_time_s = {gate}\n"
        "# columns: MJD Ch1 Ch2 Ch3 Ch4 Ch5 Ch6 Ch7 Ch8\n"
    )
    # 11 decimals of a day ~ 0.86 us resolution, enough for a 1 ms gate.
    fmt = ["%.11f"] + ["%.6f"] * 8
    with open(path, "w") as fh:
        fh.write(header)
        np.savetxt(fh, matrix, fmt=fmt, delimiter="\t")
    print(f"wrote {path}  ({n} rows, gate {gate*1e3:.3g} ms)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="duration per file in seconds (default 20)")
    ap.add_argument("--gate", type=float, default=1e-3,
                    help="gate time in seconds (default 1e-3 = 1 ms)")
    ap.add_argument("--outdir", default=os.path.dirname(__file__) or ".",
                    help="output directory")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    n = int(round(args.seconds / args.gate))
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    start = 60000.0  # arbitrary MJD (~2023)
    p1 = os.path.join(args.outdir, "fxe_beat_part1.txt")
    p2 = os.path.join(args.outdir, "fxe_beat_part2.txt")
    write_file(p1, start, n, args.gate, rng)
    # Second file starts right after the first ends.
    start2 = start + (n * args.gate) / SECONDS_PER_DAY
    write_file(p2, start2, n, args.gate, rng)


if __name__ == "__main__":
    main()
