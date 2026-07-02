# FXE Frequency Counter Analyzer

A desktop GUI to **load FXE frequency-counter files**, **plot up to 8 channels**
with interactive zoom and time-interval selection, **join several files** onto a
common time base, and **compute the fractional frequency stability** (overlapping
Allan deviation) of a beat at averaging times such as **1 ms, 10 ms, 100 ms,
200 ms, 1 s and 10 s**.

Built with Python + [pyqtgraph](https://www.pyqtgraph.org/) (fast, mouse-driven
plots that stay responsive on millions of samples) and NumPy.

---

## Features (mapped to the request)

| You asked for… | Where it is |
|---|---|
| Read the FXE counter channels | Robust text loader, auto-detects delimiter & header (`loader.py`) |
| Upload a file from the PC | **Load file(s)…** button → native file dialog |
| Choose any channel to plot | **Channels to display** check-boxes |
| Join multiple files with timestamps | **Add & join…** merges files chronologically on their timestamp |
| Plot 8 channels separately in one GUI | Up to 8 individual plots in a configurable grid |
| Display screen showing 8 plots | **Plot grid** selector (Auto / 2×2 / 2×4 / 4×2 / 8×1 …) |
| Zoom in | Mouse wheel = zoom, drag = pan, right-drag = box-zoom, right-click → *View All* |
| Select the desired time interval | Draggable **time-interval selector** shared across all plots |
| Fractional stability at 1 ms…10 s | **Fractional stability** panel: overlapping Allan deviation + log-log plot + table |

---

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `PyQt5`, `pyqtgraph` (plus `pytest` for the test suite).

## Run

```bash
python run.py
# or
python -m fxe_analyzer
```

## Try it immediately (synthetic data)

Generate two consecutive FXE-style files (8 channels, 1 ms gate) and load them:

```bash
python sample_data/generate_sample.py --seconds 20
```

This writes `sample_data/fxe_beat_part1.txt` and `…part2.txt`. In the GUI:

1. **Load file(s)…** → pick `fxe_beat_part1.txt`. The import dialog is
   pre-filled (MJD timestamp column + 8 channels); click **OK**.
2. **Add & join…** → pick `fxe_beat_part2.txt` to append it in time.
3. Toggle channels, change the plot grid, drag the blue region to pick a window.
4. Choose the τ values and click **Compute (full range)** or **Compute (selection)**.

---

## The import dialog (column mapping)

FXE / K+K-style counters export plain-text tables whose exact delimiter and
column order depend on the software version, so nothing is hard-coded. When you
load the first file, a dialog lets you confirm:

- **Timestamp column** and how to read it:
  - `datetime` — a **date column (YYMMDD)** + a **time column (HHMMSS.sss)**,
    as used by the FXE 8-channel export. Converted to absolute time.
  - `hhmmss` — a single time-of-day column (`HHMMSS.sss`).
  - `index` — a sample counter; multiplied by the **gate time**.
  - `seconds` / `unix` — absolute seconds.
  - `mjd` — Modified Julian Date (days); converted to seconds automatically.
  - *None* — no timestamp; the time axis is synthesised from the **gate time**.
- **Gate time** (sample interval, e.g. `0.001` s for a 1 ms gate). Used when
  there is no timestamp column or for `index` mode.
- **Channels** — tick up to 8 columns and name them (defaults `Ch1…Ch8`).
- **Invalid samples** — over-range / unlocked readings (the counter's
  `99999999.999` sentinel) are blanked to gaps so they don't corrupt the plots
  or the Allan deviation. You can change or disable the value.

### Recognised FXE 8-channel layout

The typical export —

```
YYMMDD  HHMMSS.sss  flag  ch1  ch2 … ch8  0.0
260630  153814.320   1    20079316.175  …            0.00000000000
```

— is auto-detected: columns 0–1 become the `datetime` timestamp, the `flag`
(0/1) and trailing all-zero columns are skipped, and columns 3–10 become the 8
channels. Just confirm the dialog and click **OK**.

The same mapping is applied to every file you join, so keep joined files in the
same export format.

### Joining files

- **Absolute timestamps** (`seconds` / `unix` / `mjd`): files are concatenated
  and **sorted chronologically**.
- **Relative time** (`index` / *None*): each new file is **appended** after the
  previous one on the time axis.

Non-uniform sample spacing (gaps) is detected and reported in the status bar —
the Allan deviation assumes uniform sampling, so mind gaps in joined records.

---

## Fractional stability (Allan deviation)

The panel computes the **overlapping Allan deviation** σ(τ), the standard
estimator of frequency stability, at each averaging time τ = m·τ₀ (τ₀ is the
gate time, m an integer). For N samples fᵢ:

```
σ²(τ) = 1 / (2 m² (N − 2m + 1)) · Σⱼ [ Σ_{i=j}^{j+m−1} (f_{i+m} − f_i) ]²
```

Two quantities are reported:

- **Absolute** σ(τ) in **Hz** — the stability of the measured (beat) frequency.
- **Fractional** σ(τ) — dimensionless, = absolute σ(τ) / **reference ν₀**.

Because the estimator uses only *differences*, it is unaffected by a constant
offset, so `σ_fractional = σ_Hz / ν₀`. Choose ν₀ with the controls:

- **Use channel mean** (default): ν₀ = mean of that channel → the stability of
  the beat *relative to itself*.
- **Reference / carrier ν₀**: enter the carrier the beat is referenced to
  (e.g. an optical frequency ~3×10¹⁴ Hz) → the true fractional stability.

τ values that are shorter than the gate time, or too long for the record (e.g.
10 s needs > 20 000 samples at a 1 ms gate), are skipped automatically. Add extra
τ values in the “extra τ” box (comma-separated).

Results appear as a **table** (τ × channel) and a **log-log σ(τ) plot**, and can
be **exported to CSV** (`channel, tau_s, m, n, adev_hz, err_hz, adev_fractional,
err_fractional`). The reported uncertainty is a simple 1σ estimate based on the
number of independent samples.

---

## Project layout

```
fxe_analyzer/
  loader.py            file parsing, column mapping, multi-file join
  stability.py         overlapping Allan deviation / fractional stability
  app.py               GUI entry point
  gui/
    main_window.py     controls, plot grid + stability panel
    import_dialog.py   column-mapping dialog
    plot_grid.py       8-plot grid, linked zoom, region selector
sample_data/
  generate_sample.py   synthetic FXE files for testing
tests/                 unit tests (loader, stability) + headless GUI smoke tests
run.py                 launcher
```

## Tests

```bash
pytest -q
```

The stability math is cross-checked against an independent brute-force Allan
implementation and against the known τ⁻¹ᐟ² slope of white frequency noise. GUI
tests run headless via Qt's `offscreen` platform.

---

## Notes & assumptions

- The loader targets **frequency** data (Hz), one column per channel, as stated
  for the FXE. If your export uses a different delimiter or layout, adjust it in
  the import dialog — the parser auto-detects commas, semicolons, tabs and
  whitespace and skips comment/header lines.
- For a clean 1 ms gate, MJD timestamps need enough decimals (~10–11) to resolve
  a millisecond; the sample generator writes 11. If your timestamps are coarse,
  prefer `index` mode with an explicit gate time so τ₀ is exact.
