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
TIME_MODES = ("index", "seconds", "unix", "mjd")


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

    def validate(self, n_cols: int) -> None:
        if self.time_mode not in TIME_MODES:
            raise ValueError(f"unknown time_mode {self.time_mode!r}")
        if self.gate_time <= 0:
            raise ValueError("gate_time must be positive")
        if self.time_col is not None and not (0 <= self.time_col < n_cols):
            raise ValueError(f"time_col {self.time_col} out of range")
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


def suggest_mapping(raw: RawFile, max_channels: int = 8) -> ColumnMapping:
    """Heuristically propose a column mapping for a parsed file.

    Assumes the first column is a timestamp when it is monotonically
    increasing and clearly different in scale from the others; otherwise
    treats every column as a channel.
    """
    n_cols = raw.n_cols
    channel_cols: Dict[str, int] = {}
    time_col: Optional[int] = None
    time_mode = "index"

    if n_cols == 0:
        return ColumnMapping(channel_cols={"Ch1": 0})

    first = raw.data[:, 0]
    monotonic = np.all(np.diff(first) > 0) if first.size > 1 else False
    if monotonic and n_cols >= 2:
        time_col = 0
        med = float(np.median(first))
        if 40000 < med < 80000:      # plausible MJD (covers ~1968–2077)
            time_mode = "mjd"
        elif med > 1e8:              # large -> Unix epoch seconds
            time_mode = "unix"
        else:
            time_mode = "seconds"
        data_cols = list(range(1, n_cols))
    else:
        data_cols = list(range(n_cols))

    for i, col in enumerate(data_cols[:max_channels], start=1):
        channel_cols[f"Ch{i}"] = col

    return ColumnMapping(
        channel_cols=channel_cols,
        time_col=time_col,
        time_mode=time_mode,
        gate_time=1e-3,
    )


def _time_axis(raw: RawFile, mapping: ColumnMapping) -> tuple[np.ndarray, bool]:
    """Return (time_seconds, absolute_time) for a single file."""
    n = raw.n_rows
    if mapping.time_col is None or mapping.time_mode == "index":
        if mapping.time_col is not None and mapping.time_mode == "index":
            idx = raw.data[:, mapping.time_col]
            return idx * mapping.gate_time, False
        return np.arange(n) * mapping.gate_time, False

    col = raw.data[:, mapping.time_col]
    if mapping.time_mode == "seconds":
        return col.astype(float), True
    if mapping.time_mode == "unix":
        return col.astype(float), True
    if mapping.time_mode == "mjd":
        # Convert MJD (days) to Unix epoch seconds.
        return (col - _MJD_UNIX_EPOCH) * SECONDS_PER_DAY, True
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
            chan_data[name].append(raw.data[:, col].astype(float))

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
