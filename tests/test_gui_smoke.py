"""Headless smoke test for the GUI stack.

Runs Qt with the 'offscreen' platform so it works without a display.
Skips gracefully if PyQt5/pyqtgraph are not installed.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5 import QtWidgets  # noqa: E402

from fxe_analyzer import loader, stability  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_files(tmp_path):
    """Two consecutive 3 s files at 1 ms gate, MJD timestamp + 2 channels."""
    paths = []
    n = 3000
    for k in range(2):
        mjd = 60000.0 + (k * n + np.arange(n)) * 1e-3 / 86400.0
        ch1 = 1.0e6 + np.random.default_rng(k).standard_normal(n)
        ch2 = 2.0e6 + np.random.default_rng(k + 10).standard_normal(n)
        p = tmp_path / f"f{k}.txt"
        np.savetxt(p, np.column_stack([mjd, ch1, ch2]),
                   fmt=["%.11f", "%.4f", "%.4f"])
        paths.append(str(p))
    return paths


def test_gui_load_plot_compute(app, tmp_path):
    from fxe_analyzer.gui.main_window import MainWindow

    paths = _make_files(tmp_path)
    mw = MainWindow()

    mw.paths = list(paths)
    first = loader.parse_file(paths[0])
    mw.mapping = loader.suggest_mapping(first)
    assert mw.mapping.time_mode == "mjd"
    mw._build_dataset()

    ds = mw.dataset
    assert ds.time.size == 6000
    assert ds.channel_names == ["Ch1", "Ch2"]
    assert len(mw.plot_grid._plots) == 2
    assert len(mw.plot_grid._regions) == 2

    # Region selection -> mask covers ~2 s.
    mw.plot_grid.set_region(0.0, 2.0)
    mask = mw.plot_grid.selection_mask("Ch1")
    assert 1950 < int(mask.sum()) < 2100

    # Compute over full range at the requested averaging times.
    mw._compute(False)
    assert set(mw.last_results) == {"Ch1", "Ch2"}
    taus = sorted(round(p.tau, 6) for p in mw.last_results["Ch1"])
    assert 1e-3 in taus
    # tau rows + 3 summary rows (slope, noise type, drift).
    assert mw.result_table.rowCount() == len(taus) + 3
    assert mw.result_table.columnCount() == 2
    # Per-channel summary populated.
    assert set(mw.last_summary) == {"Ch1", "Ch2"}
    slope, noise, drift = mw.last_summary["Ch1"]
    assert isinstance(noise, str)

    # Compute over a 2 s selection -> long taus are dropped.
    mw._compute(True)
    taus_sel = {round(p.tau, 6) for p in mw.last_results["Ch1"]}
    assert 10.0 not in taus_sel


def test_gui_export_images_and_detrend(app, tmp_path):
    from fxe_analyzer.gui.main_window import MainWindow

    paths = _make_files(tmp_path)
    mw = MainWindow()
    mw.paths = list(paths)
    mw.mapping = loader.suggest_mapping(loader.parse_file(paths[0]))
    mw._build_dataset()

    # Drift removal option feeds through to compute.
    mw.detrend_combo.setCurrentIndex(1)  # linear
    mw._compute(False)
    assert mw.last_summary  # slope/noise/drift computed

    # Export channel plots and the Allan plot to PNG.
    png1 = str(tmp_path / "channels.png")
    mw.plot_grid.export_image(png1)
    assert os.path.exists(png1) and os.path.getsize(png1) > 0

    png2 = str(tmp_path / "allan.png")
    mw.adev_plot.grab().save(png2)
    assert os.path.exists(png2) and os.path.getsize(png2) > 0


def test_gui_layout_and_channel_toggle(app, tmp_path):
    from fxe_analyzer.gui.main_window import MainWindow

    paths = _make_files(tmp_path)
    mw = MainWindow()
    mw.paths = list(paths)
    mw.mapping = loader.suggest_mapping(loader.parse_file(paths[0]))
    mw._build_dataset()

    mw.channel_checks["Ch2"].setChecked(False)
    assert mw._selected_channels() == ["Ch1"]
    assert len(mw.plot_grid._plots) == 1

    mw.layout_combo.setCurrentText("2 × 1")
    mw.channel_checks["Ch2"].setChecked(True)
    assert len(mw.plot_grid._plots) == 2
