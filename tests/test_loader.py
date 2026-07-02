"""Tests for the FXE file loader and multi-file join logic."""

import numpy as np
import pytest

from fxe_analyzer import loader


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_parse_whitespace_with_header(tmp_path):
    text = (
        "# FXE export\n"
        "# columns: idx Ch1 Ch2\n"
        "0  100.0  200.0\n"
        "1  100.1  200.2\n"
        "2  100.2  200.4\n"
    )
    path = _write(tmp_path, "a.txt", text)
    raw = loader.parse_file(path)
    assert raw.n_rows == 3
    assert raw.n_cols == 3
    assert len(raw.header_lines) == 2
    assert raw.data[1, 1] == pytest.approx(100.1)


def test_sniff_semicolon(tmp_path):
    text = "10.0;1.0;2.0\n11.0;1.1;2.1\n12.0;1.2;2.2\n"
    path = _write(tmp_path, "b.csv", text)
    raw = loader.parse_file(path)
    assert raw.delimiter == ";"
    assert raw.n_cols == 3


def test_sniff_comma(tmp_path):
    text = "0,5.0,6.0,7.0\n1,5.1,6.1,7.1\n"
    path = _write(tmp_path, "c.csv", text)
    raw = loader.parse_file(path)
    assert raw.delimiter == ","
    assert raw.n_cols == 4


def test_suggest_mapping_detects_mjd_time(tmp_path):
    # First column looks like MJD (monotonic, ~60000).
    rows = "\n".join(
        f"{60000.0 + i/86400.0:.9f}\t{1.0+i}\t{2.0+i}" for i in range(5)
    )
    path = _write(tmp_path, "d.txt", rows + "\n")
    raw = loader.parse_file(path)
    mapping = loader.suggest_mapping(raw)
    assert mapping.time_col == 0
    assert mapping.time_mode == "mjd"
    assert list(mapping.channel_cols.values()) == [1, 2]


def test_suggest_mapping_no_time_column(tmp_path):
    # Non-monotonic first column -> everything is a channel.
    text = "5.0 1.0\n3.0 2.0\n9.0 3.0\n"
    path = _write(tmp_path, "e.txt", text)
    raw = loader.parse_file(path)
    mapping = loader.suggest_mapping(raw)
    assert mapping.time_col is None
    assert len(mapping.channel_cols) == 2


def test_build_dataset_synth_time(tmp_path):
    text = "100.0\n101.0\n102.0\n103.0\n"
    path = _write(tmp_path, "f.txt", text)
    raw = loader.parse_file(path)
    mapping = loader.ColumnMapping(
        channel_cols={"Ch1": 0}, time_col=None, gate_time=1e-3
    )
    ds = loader.build_dataset([raw], mapping)
    assert ds.tau0 == pytest.approx(1e-3)
    assert ds.time[1] == pytest.approx(1e-3)
    assert not ds.absolute_time
    np.testing.assert_allclose(ds.channels["Ch1"], [100, 101, 102, 103])


def test_join_relative_files_appended(tmp_path):
    t1 = "0.0 10.0\n0.001 11.0\n0.002 12.0\n"
    t2 = "0.0 20.0\n0.001 21.0\n0.002 22.0\n"
    p1 = _write(tmp_path, "g1.txt", t1)
    p2 = _write(tmp_path, "g2.txt", t2)
    mapping = loader.ColumnMapping(
        channel_cols={"Ch1": 1}, time_col=0, time_mode="seconds"
    )
    ds = loader.build_dataset([p1, p2], mapping, join=True)
    # Absolute time here means the second file is sorted/merged by value,
    # but because both start at 0 with 'seconds' mode we force relative
    # append only when not absolute; 'seconds' is treated as absolute, so
    # verify chronological sort keeps all 6 samples.
    assert ds.time.size == 6
    assert ds.channels["Ch1"].size == 6


def test_join_index_mode_offsets(tmp_path):
    # 'index' time_mode is relative: files should be concatenated in order.
    t1 = "0 10.0\n1 11.0\n2 12.0\n"
    t2 = "0 20.0\n1 21.0\n2 22.0\n"
    p1 = _write(tmp_path, "h1.txt", t1)
    p2 = _write(tmp_path, "h2.txt", t2)
    mapping = loader.ColumnMapping(
        channel_cols={"Ch1": 1}, time_col=0, time_mode="index", gate_time=1e-3
    )
    ds = loader.build_dataset([p1, p2], mapping, join=True)
    assert ds.channels["Ch1"].tolist() == [10, 11, 12, 20, 21, 22]
    # time must be strictly increasing across the join
    assert np.all(np.diff(ds.time) > 0)


def test_mjd_converted_to_seconds(tmp_path):
    rows = "\n".join(
        f"{60000.0 + i/86400.0:.9f} {1.0+i}" for i in range(4)
    )
    path = _write(tmp_path, "i.txt", rows + "\n")
    mapping = loader.ColumnMapping(
        channel_cols={"Ch1": 1}, time_col=0, time_mode="mjd"
    )
    ds = loader.build_dataset([path], mapping)
    assert ds.absolute_time
    # 1 s spacing in MJD -> ~1 s spacing in seconds. (Tolerance is loose
    # because the test data's MJD values are only written to 9 decimals;
    # what matters is that the day->second *86400 conversion happened, i.e.
    # tau0 is ~1 and not ~1e-5.)
    assert ds.tau0 == pytest.approx(1.0, abs=1e-3)


def test_gap_detection(tmp_path):
    text = "0.0 1\n0.001 2\n0.002 3\n0.010 4\n0.011 5\n"  # jump at index 3
    path = _write(tmp_path, "j.txt", text)
    mapping = loader.ColumnMapping(
        channel_cols={"Ch1": 1}, time_col=0, time_mode="seconds"
    )
    ds = loader.build_dataset([path], mapping)
    assert 2 in ds.gap_indices  # gap starts after sample index 2


# A few real lines of the FXE "YYMMDD HHMMSS.sss flag ch1..ch8 zero" export,
# including the 99999999.999 over-range sentinel on some channels.
_FXE_SAMPLE = (
    "260630 153814.320 1  20079316.17539099980  26478291.56656499950  49999992.97996800390  37366052.71031400560  21646228.27099500220  41282112.55396899580  28217775.18150600050  27520317.63587700200         0.00000000000\n"
    "260630 153814.430 0  20079316.17507900300  99999999.99899999800  49999992.89527599510  37370623.14537499840  21645653.79367100070  41282111.44092500210  26969838.45996100080  29512635.00732345880         0.00000000000\n"
    "260630 153814.524 0  20079316.17387099940  99999999.99899999800  49999992.89879199860  37375741.09910999980  21645013.55651900170  41282110.69635400180  99999999.99899999800  99999999.99899999800         0.00000000000\n"
    "260630 153814.632 0  20079316.17475000020  40423876.14490629730  49999992.92098400000  37376646.98231200130  21644806.68730999900  41282109.76968800280  43327558.86247767510  30469387.60637113820         0.00000000000\n"
)


def test_parse_real_fxe_format(tmp_path):
    path = _write(tmp_path, "fxe.txt", _FXE_SAMPLE)
    raw = loader.parse_file(path)
    assert raw.n_cols == 12
    assert raw.n_rows == 4

    m = loader.suggest_mapping(raw)
    assert m.time_mode == "datetime"
    assert m.date_col == 0 and m.time_col == 1
    # Flag column (2) and trailing all-zero column (11) are skipped.
    assert m.channel_cols["Ch1"] == 3
    assert 2 not in m.channel_cols.values()
    assert 11 not in m.channel_cols.values()
    assert len(m.channel_cols) == 8

    ds = loader.build_dataset([raw], m)
    assert ds.absolute_time
    # Timestamp of the first row: 2026-06-30 15:38:14.320 UTC.
    expected = (
        float((np.datetime64("2026-06-30") - np.datetime64("1970-01-01"))
              / np.timedelta64(1, "s"))
        + 15 * 3600 + 38 * 60 + 14.320
    )
    assert ds.time[0] == pytest.approx(expected, abs=1e-3)
    # Over-range sentinels on Ch2 (rows 1 & 2) become NaN; Ch1 is clean.
    assert np.isnan(ds.channels["Ch2"]).sum() == 2
    assert not np.any(np.isnan(ds.channels["Ch1"]))
    assert ds.channels["Ch1"][0] == pytest.approx(20079316.175, abs=1e-2)


def test_invalid_value_disabled(tmp_path):
    path = _write(tmp_path, "fxe2.txt", _FXE_SAMPLE)
    raw = loader.parse_file(path)
    m = loader.suggest_mapping(raw)
    m.invalid_value = None  # keep the raw sentinel values
    ds = loader.build_dataset([raw], m)
    assert not np.any(np.isnan(ds.channels["Ch2"]))
    assert ds.channels["Ch2"][1] == pytest.approx(99999999.999, abs=1e-2)


def test_hhmmss_time_only(tmp_path):
    text = "153814.000 10.0\n153815.000 11.0\n153816.000 12.0\n"
    path = _write(tmp_path, "t.txt", text)
    m = loader.ColumnMapping(
        channel_cols={"Ch1": 1}, time_col=0, time_mode="hhmmss"
    )
    ds = loader.build_dataset([path], m)
    assert ds.absolute_time
    assert ds.tau0 == pytest.approx(1.0, abs=1e-6)
    assert ds.time[0] == pytest.approx(15 * 3600 + 38 * 60 + 14.0, abs=1e-6)


def test_datetime_requires_date_col(tmp_path):
    text = "153814.0 10.0\n153815.0 11.0\n"
    path = _write(tmp_path, "u.txt", text)
    raw = loader.parse_file(path)
    with pytest.raises(ValueError):
        loader.ColumnMapping(
            channel_cols={"Ch1": 1}, time_col=0, time_mode="datetime"
        ).validate(raw.n_cols)


def test_mapping_validation_errors(tmp_path):
    text = "1 2\n3 4\n"
    path = _write(tmp_path, "k.txt", text)
    raw = loader.parse_file(path)
    with pytest.raises(ValueError):
        loader.ColumnMapping(channel_cols={"Ch1": 5}).validate(raw.n_cols)
    with pytest.raises(ValueError):
        loader.ColumnMapping(channel_cols={}, ).validate(raw.n_cols)
