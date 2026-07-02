"""Loading and joining FXE frequency-counter files.

FXE-style counters (e.g. K+K Messtechnik multi-channel counters) export
plain-text tables: an optional header/metadata block followed by numeric
rows, one row per gate interval, with one column per channel and often a
leading timestamp column.  Exact delimiters and column layouts vary with
software version and configuration, so this loader is deliberately
tolerant:

* it auto-detects the delimiter (comma / semicolon / tab / whitespace),
* it skips non-numeric header/comment lines (keeping them as metadata),
* it lets the caller map columns to a timestamp and to channels, and
* it can join several files onto a single, sorted time base.

The timestamp column may be interpreted as relative seconds, absolute
Unix epoch seconds, Modified Julian Date (days), or a plain sample index
(in which case a gate time supplies the spacing).  When no timestamp
column is available the time axis is synthesised from the gate time.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Candidate delimiters, tried in this order.
_DELIMITERS = [",", ";", "\t", None]  # None => any run of whitespace
_MJD_UNIX_EPOCH = 40587.0  # MJD of 1970-01-01, for MJD -> Unix conversion
SECONDS_PER_DAY = 86400.0

# Recognised timestamp interpretations.
#   index    : sample counter (x gate time)
#   seconds  : absolute seconds
#   unix     : absolute Unix epoch seconds
#   mjd      : Modified Julian Date (days)
#   datetime : a YYMMDD date column + an HHMMSS.sss time column
#   hhmmss   : an HHMMSS.sss time-of-day column (no date)
TIME_MODES = ("index", "seconds", "unix", "mjd", "datetime", "hhmmss")

# Value an FXE/K+K counter reports for an over-range / unlocked channel.
DEFAULT_INVALID_VALUE = 99999999.999


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _split(line: str, delim: Optional[str]) -> List[str]:
    if delim is None:
        return line.split()
    return [p.strip() for p in line.split(delim)]


@dataclass
class RawFile:
    """A parsed numeric table plus the header lines that preceded it."""

    path: str
    delimiter: Optional[str]
    header_lines: List[str]
    data: np.ndarray  # shape (n_rows, n_cols)

    @property
    def n_cols(self) -> int:
        return 0 if self.data.size == 0 else self.data.shape[1]

    @property
    def n_rows(self) -> int:
        return self.data.shape[0] if self.data.ndim == 2 else 0


@dataclass
class ColumnMapping:
    """How to turn a RawFile's columns into a time axis and channels."""

    channel_cols: Dict[str, int]           # channel name -> column index
    time_col: Optional[int] = None         # column index or None (synth)
    time_mode: str = "index"               # one of TIME_MODES
    gate_time: float = 1e-3                # used for 'index' / synthesised time
    date_col: Optional[int] = None         # YYMMDD column (for 'datetime' mode)
    invalid_value: Optional[float] = DEFAULT_INVALID_VALUE  # -> NaN, None disables
    invalid_tol: float = 1e-2              # match tolerance for invalid_value

    def validate(self, n_cols: int) -> None:
        if self.time_mode not in TIME_MODES:
            raise ValueError(f"unknown time_mode {self.time_mode!r}")
        if self.gate_time <= 0:
            raise ValueError("gate_time must be positive")
        if self.time_col is not None and not (0 <= self.time_col < n_cols):
            raise ValueError(f"time_col {self.time_col} out of range")
        if self.time_mode in ("datetime", "hhmmss") and self.time_col is None:
            raise ValueError(f"{self.time_mode} mode needs a time column")
        if self.time_mode == "datetime":
            if self.date_col is None or not (0 <= self.date_col < n_cols):
                raise ValueError("datetime mode needs a valid date column")
        if not self.channel_cols:
            raise ValueError("no channels selected")
        for name, col in self.channel_cols.items():
            if not (0 <= col < n_cols):
                raise ValueError(f"channel {name!r} column {col} out of range")


@dataclass
class Dataset:
    """A time base plus one array per channel, ready to plot/analyse."""

    time: np.ndarray                       # seconds (relative or absolute)
    channels: Dict[str, np.ndarray]        # name -> frequency samples (Hz)
    tau0: float                            # median sample spacing (s)
    absolute_time: bool                    # True if `time` is absolute epoch
    source_files: List[str] = field(default_factory=list)
    gap_indices: List[int] = field(default_factory=list)  # i where a gap starts
    header_lines: List[str] = field(default_factory=list)

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels.keys())

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.time.size else 0.0


def sniff_delimiter(lines: Sequence[str]) -> Optional[str]:
    """Pick the delimiter that yields the most consistent numeric columns."""
    best_delim: Optional[str] = None
    best_score = -1
    for delim in _DELIMITERS:
        counts = []
        for line in lines:
            parts = _split(line, delim)
            if len(parts) >= 2 and all(_is_number(p) for p in parts if p != ""):
                counts.append(len([p for p in parts if p != ""]))
        if not counts:
            continue
        # Reward many rows with a consistent, wide column count.
        mode = max(set(counts), key=counts.count)
        consistent = sum(1 for c in counts if c == mode)
        score = consistent * 100 + mode
        if score > best_score:
            best_score = score
            best_delim = delim
    return best_delim


def parse_file(path: str, delimiter: Optional[str] = "__auto__") -> RawFile:
    """Read a file into a numeric matrix, keeping leading header lines.

    ``delimiter`` may be an explicit delimiter, ``None`` for whitespace,
    or the sentinel ``"__auto__"`` to auto-detect.
    """
    with open(path, "r", errors="replace") as fh:
        text_lines = fh.read().splitlines()

    non_empty = [ln for ln in text_lines if ln.strip() != ""]
    if delimiter == "__auto__":
        # Sniff using a sample of the later lines (skip a possible header).
        sample = non_empty[-200:] if len(non_empty) > 200 else non_empty
        delimiter = sniff_delimiter(sample)

    header_lines: List[str] = []
    rows: List[List[float]] = []
    for ln in text_lines:
        if ln.strip() == "":
            continue
        parts = [p for p in _split(ln, delimiter) if p != ""]
        if len(parts) >= 1 and all(_is_number(p) for p in parts):
            rows.append([float(p) for p in parts])
        else:
            # Header/metadata (or a stray annotation between data rows).
            if not rows:
                header_lines.append(ln)

    if not rows:
        data = np.empty((0, 0))
    else:
        width = max(len(r) for r in rows)
        # Keep only rows with the dominant width (guards against ragged tails).
        widths = [len(r) for r in rows]
        dom = max(set(widths), key=widths.count)
        data = np.array([r for r in rows if len(r) == dom], dtype=float)
        _ = width  # (kept for clarity; dom is what we use)

    return RawFile(
        path=path, delimiter=delimiter, header_lines=header_lines, data=data
    )


def _is_integer_col(col: np.ndarray) -> bool:
    return bool(np.all(np.abs(col - np.round(col)) < 1e-6))


def _looks_like_yymmdd(col: np.ndarray) -> bool:
    """6-digit YYMMDD integers with valid month/day."""
    if col.size == 0 or not _is_integer_col(col):
        return False
    iv = np.round(col).astype(np.int64)
    if np.median(iv) < 10101 or np.max(iv) > 999999 or np.min(iv) < 0:
        return False
    mm = (iv // 100) % 100
    dd = iv % 100
    return bool(np.all((mm >= 1) & (mm <= 12) & (dd >= 1) & (dd <= 31)))


def _looks_like_hhmmss(col: np.ndarray) -> bool:
    """HHMMSS.sss time-of-day values (00:00:00 .. 23:59:59.xxx)."""
    if col.size == 0 or np.min(col) < 0 or np.max(col) >= 240000:
        return False
    hh = np.floor(col / 10000.0)
    rem = col - hh * 10000.0
    mm = np.floor(rem / 100.0)
    ss = rem - mm * 100.0
    return bool(np.all((hh <= 23) & (mm <= 59) & (ss < 60)))


def _is_binary_flag(col: np.ndarray) -> bool:
    if not _is_integer_col(col):
        return False
    return set(np.unique(np.round(col).astype(int)).tolist()).issubset({0, 1})


def _is_all_zero(col: np.ndarray) -> bool:
    return bool(np.all(col == 0.0))


def suggest_mapping(raw: RawFile, max_channels: int = 8) -> ColumnMapping:
    """Heuristically propose a column mapping for a parsed file.

    Recognises three common shapes, in order of preference:

    1. ``YYMMDD HHMMSS.sss flag  ch1..chN  [zeros]`` (the FXE/K+K export):
       columns 0 and 1 become a ``datetime`` timestamp and constant
       flag / all-zero columns are skipped when picking channels.
    2. A single leading, monotonically-increasing timestamp column
       (interpreted as MJD / Unix / seconds by its magnitude).
    3. No timestamp — every column is treated as a channel.
    """
    n_cols = raw.n_cols
    if n_cols == 0:
        return ColumnMapping(channel_cols={"Ch1": 0})

    col0 = raw.data[:, 0]
    channel_cols: Dict[str, int] = {}
    date_col: Optional[int] = None
    time_col: Optional[int] = None
    time_mode = "index"

    # 1) date + time-of-day pair (unambiguous: needs a YYMMDD integer column).
    if n_cols >= 4 and _looks_like_yymmdd(col0) and _looks_like_hhmmss(raw.data[:, 1]):
        date_col, time_col, time_mode = 0, 1, "datetime"
        candidates = list(range(2, n_cols))
    # 2) monotonic numeric timestamp (MJD / Unix / seconds by magnitude).
    #    (A standalone HHMMSS column is not auto-detected because its value
    #    range overlaps MJD/seconds; pick 'hhmmss' manually if needed.)
    elif col0.size > 1 and np.all(np.diff(col0) > 0) and n_cols >= 2:
        time_col = 0
        med = float(np.median(col0))
        if 40000 < med < 80000:
            time_mode = "mjd"
        elif med > 1e8:
            time_mode = "unix"
        else:
            time_mode = "seconds"
        candidates = list(range(1, n_cols))
    else:
        candidates = list(range(n_cols))

    # Skip flag (0/1) and all-zero columns when choosing channels.
    channel_candidates = [
        c for c in candidates
        if not _is_binary_flag(raw.data[:, c]) and not _is_all_zero(raw.data[:, c])
    ] or candidates  # fall back to raw candidates if the filter emptied it

    for i, col in enumerate(channel_candidates[:max_channels], start=1):
        channel_cols[f"Ch{i}"] = col

    return ColumnMapping(
        channel_cols=channel_cols,
        time_col=time_col,
        time_mode=time_mode,
        gate_time=1e-3,
        date_col=date_col,
    )


def _seconds_of_day(time_col: np.ndarray) -> np.ndarray:
    """Convert an HHMMSS.sss column into seconds since midnight."""
    t = time_col.astype(float)
    hh = np.floor(t / 10000.0)
    rem = t - hh * 10000.0
    mm = np.floor(rem / 100.0)
    ss = rem - mm * 100.0
    return hh * 3600.0 + mm * 60.0 + ss


def _datetime_to_epoch(date_col: np.ndarray, time_col: np.ndarray) -> np.ndarray:
    """Combine a YYMMDD date column and HHMMSS.sss time column into epoch s."""
    sod = _seconds_of_day(time_col)
    date_int = np.round(date_col).astype(np.int64)
    base = {}
    for dv in np.unique(date_int):
        y = 2000 + dv // 10000
        m = (dv // 100) % 100
        d = dv % 100
        dt = np.datetime64(f"{int(y):04d}-{int(m):02d}-{int(d):02d}")
        base[int(dv)] = float(
            (dt - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
        )
    day_start = np.array([base[int(dv)] for dv in date_int], dtype=float)
    return day_start + sod


def _time_axis(raw: RawFile, mapping: ColumnMapping) -> tuple[np.ndarray, bool]:
    """Return (time_seconds, absolute_time) for a single file."""
    n = raw.n_rows
    if mapping.time_col is None or mapping.time_mode == "index":
        if mapping.time_col is not None and mapping.time_mode == "index":
            idx = raw.data[:, mapping.time_col]
            return idx * mapping.gate_time, False
        return np.arange(n) * mapping.gate_time, False

    col = raw.data[:, mapping.time_col]
    if mapping.time_mode in ("seconds", "unix"):
        return col.astype(float), True
    if mapping.time_mode == "mjd":
        # Convert MJD (days) to Unix epoch seconds.
        return (col - _MJD_UNIX_EPOCH) * SECONDS_PER_DAY, True
    if mapping.time_mode == "hhmmss":
        return _seconds_of_day(col), True
    if mapping.time_mode == "datetime":
        return _datetime_to_epoch(raw.data[:, mapping.date_col], col), True
    raise ValueError(f"unknown time_mode {mapping.time_mode!r}")


def build_dataset(
    files: Sequence[str] | Sequence[RawFile],
    mapping: ColumnMapping,
    join: bool = True,
) -> Dataset:
    """Load one or more files and assemble a joined :class:`Dataset`.

    Files are parsed with the same ``mapping``.  When they carry absolute
    timestamps they are merged and sorted chronologically; otherwise each
    subsequent file is appended after the previous one on the relative
    time axis.  Sample-spacing outliers are recorded in ``gap_indices``.
    """
    raws: List[RawFile] = []
    for f in files:
        raws.append(f if isinstance(f, RawFile) else parse_file(f))
    if not raws:
        raise ValueError("no files to load")

    mapping.validate(raws[0].n_cols)

    times: List[np.ndarray] = []
    chan_data: Dict[str, List[np.ndarray]] = {n: [] for n in mapping.channel_cols}
    absolute = False
    header = list(raws[0].header_lines)
    running_offset = 0.0

    for k, raw in enumerate(raws):
        if raw.n_rows == 0:
            continue
        mapping.validate(raw.n_cols)
        t, is_abs = _time_axis(raw, mapping)
        absolute = absolute or is_abs
        if not is_abs and join and k > 0:
            # Continue the relative time axis after the previous file.
            step = np.median(np.diff(t)) if t.size > 1 else mapping.gate_time
            t = t - t[0] + running_offset + step
        if not is_abs:
            running_offset = float(t[-1]) if t.size else running_offset
        times.append(t)
        for name, col in mapping.channel_cols.items():
            vals = raw.data[:, col].astype(float)
            if mapping.invalid_value is not None:
                # Blank out over-range / unlocked samples (e.g. 99999999.999).
                bad = np.abs(vals - mapping.invalid_value) <= mapping.invalid_tol
                if np.any(bad):
                    vals = vals.copy()
                    vals[bad] = np.nan
            chan_data[name].append(vals)

    if not times:
        raise ValueError("no data rows found in the provided files")

    time = np.concatenate(times)
    channels = {n: np.concatenate(v) for n, v in chan_data.items()}

    if join and absolute:
        order = np.argsort(time, kind="stable")
        time = time[order]
        channels = {n: v[order] for n, v in channels.items()}

    # Estimate the nominal gate time and flag gaps.
    if time.size > 1:
        diffs = np.diff(time)
        tau0 = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else mapping.gate_time
        gap_indices = [int(i) for i in np.where(diffs > 1.5 * tau0)[0]]
    else:
        tau0 = mapping.gate_time
        gap_indices = []

    return Dataset(
        time=time,
        channels=channels,
        tau0=tau0,
        absolute_time=absolute,
        source_files=[r.path for r in raws],
        gap_indices=gap_indices,
        header_lines=header,
    )


def load_files(
    paths: Sequence[str],
    mapping: Optional[ColumnMapping] = None,
    join: bool = True,
) -> Dataset:
    """Convenience: parse ``paths``; if no mapping is given, infer one."""
    raws = [parse_file(p) for p in paths]
    if mapping is None:
        mapping = suggest_mapping(raws[0])
    return build_dataset(raws, mapping, join=join)


def preview_text(path: str, n_lines: int = 25) -> str:
    """Return the first ``n_lines`` of a file for display in the UI."""
    out = io.StringIO()
    with open(path, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n_lines:
                out.write("...\n")
                break
            out.write(line)
    return out.getvalue()


def _basename(path: str) -> str:
    return os.path.basename(path)
