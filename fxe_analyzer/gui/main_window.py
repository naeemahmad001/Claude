"""Main application window for the FXE analyzer."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from .. import loader, stability
from ..loader import ColumnMapping, Dataset
from .import_dialog import ImportDialog
from .plot_grid import PlotGrid, auto_layout

# Averaging times offered as quick check-boxes (seconds, label).
_TAU_PRESETS = [
    ("1 ms", 1e-3), ("10 ms", 10e-3), ("100 ms", 100e-3),
    ("200 ms", 200e-3), ("1 s", 1.0), ("10 s", 10.0),
]

_LAYOUTS = {
    "Auto": None, "1 × 1": (1, 1), "1 × 2": (1, 2), "2 × 1": (2, 1),
    "2 × 2": (2, 2), "2 × 3": (2, 3), "2 × 4": (2, 4),
    "4 × 2": (4, 2), "8 × 1": (8, 1),
}

_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#72B7B2", "#EECA3B", "#B279A2", "#FF9DA6",
]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FXE Frequency Counter Analyzer")
        self.resize(1400, 900)

        self.paths: List[str] = []
        self.mapping: Optional[ColumnMapping] = None
        self.dataset: Optional[Dataset] = None
        self.channel_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self.last_results: Dict[str, list] = {}
        self.result_mode = "fractional"  # or "hz"

        self._build_ui()
        self.statusBar().showMessage("Load one or more FXE files to begin.")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Central: vertical splitter with plots on top, stability below.
        self.plot_grid = PlotGrid()
        self.stability_panel = self._build_stability_panel()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.plot_grid)
        splitter.addWidget(self.stability_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        self.plot_grid.regionChanged.connect(self._on_region_changed)

        # Left dock: controls.
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        dock.setWidget(self._build_controls())
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(320)
        v = QtWidgets.QVBoxLayout(w)

        # --- files -------------------------------------------------------
        files_box = QtWidgets.QGroupBox("Files")
        fv = QtWidgets.QVBoxLayout(files_box)
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load file(s)…")
        self.btn_add = QtWidgets.QPushButton("Add & join…")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_load.clicked.connect(lambda: self._load_files(append=False))
        self.btn_add.clicked.connect(lambda: self._load_files(append=True))
        self.btn_clear.clicked.connect(self._clear_files)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_clear)
        fv.addLayout(btn_row)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setMaximumHeight(90)
        fv.addWidget(self.file_list)
        v.addWidget(files_box)

        # --- channels ----------------------------------------------------
        self.chan_box = QtWidgets.QGroupBox("Channels to display")
        self.chan_layout = QtWidgets.QVBoxLayout(self.chan_box)
        v.addWidget(self.chan_box)

        # --- plot layout -------------------------------------------------
        disp_box = QtWidgets.QGroupBox("Display")
        dform = QtWidgets.QFormLayout(disp_box)
        self.layout_combo = QtWidgets.QComboBox()
        self.layout_combo.addItems(list(_LAYOUTS.keys()))
        self.layout_combo.currentIndexChanged.connect(self._rebuild_plots)
        dform.addRow("Plot grid:", self.layout_combo)
        self.link_check = QtWidgets.QCheckBox("Link x-axes (zoom together)")
        self.link_check.setChecked(True)
        self.link_check.toggled.connect(self.plot_grid.set_link_x)
        dform.addRow(self.link_check)
        self.region_check = QtWidgets.QCheckBox("Show time-interval selector")
        self.region_check.setChecked(True)
        self.region_check.toggled.connect(self.plot_grid.set_region_enabled)
        dform.addRow(self.region_check)
        self.btn_reset_region = QtWidgets.QPushButton("Reset selection to full range")
        self.btn_reset_region.clicked.connect(self.plot_grid.reset_region)
        dform.addRow(self.btn_reset_region)
        self.region_label = QtWidgets.QLabel("selection: —")
        self.region_label.setStyleSheet("color: gray;")
        dform.addRow(self.region_label)
        v.addWidget(disp_box)

        # --- stability ---------------------------------------------------
        stab_box = QtWidgets.QGroupBox("Fractional stability (Allan deviation)")
        sform = QtWidgets.QVBoxLayout(stab_box)
        sform.addWidget(QtWidgets.QLabel("Averaging times τ:"))
        self.tau_checks = {}
        grid = QtWidgets.QGridLayout()
        for i, (label, val) in enumerate(_TAU_PRESETS):
            chk = QtWidgets.QCheckBox(label)
            chk.setChecked(True)
            self.tau_checks[val] = chk
            grid.addWidget(chk, i // 3, i % 3)
        sform.addLayout(grid)
        self.custom_taus = QtWidgets.QLineEdit()
        self.custom_taus.setPlaceholderText("extra τ (s), comma-separated e.g. 0.5, 2, 5")
        sform.addWidget(self.custom_taus)

        ref_form = QtWidgets.QFormLayout()
        self.use_mean_ref = QtWidgets.QCheckBox("Use channel mean as reference")
        self.use_mean_ref.setChecked(True)
        self.use_mean_ref.toggled.connect(self._toggle_ref)
        ref_form.addRow(self.use_mean_ref)
        self.ref_spin = QtWidgets.QDoubleSpinBox()
        self.ref_spin.setDecimals(3)
        self.ref_spin.setRange(1.0, 1e18)
        self.ref_spin.setValue(1.0e6)
        self.ref_spin.setSuffix(" Hz")
        self.ref_spin.setEnabled(False)
        ref_form.addRow("Reference / carrier ν₀:", self.ref_spin)
        sform.addLayout(ref_form)

        self.result_mode_combo = QtWidgets.QComboBox()
        self.result_mode_combo.addItem("Fractional (dimensionless)", "fractional")
        self.result_mode_combo.addItem("Absolute (Hz)", "hz")
        self.result_mode_combo.currentIndexChanged.connect(self._refresh_results_view)
        sform.addWidget(self.result_mode_combo)

        crow = QtWidgets.QHBoxLayout()
        self.btn_calc_full = QtWidgets.QPushButton("Compute (full range)")
        self.btn_calc_sel = QtWidgets.QPushButton("Compute (selection)")
        self.btn_calc_full.clicked.connect(lambda: self._compute(False))
        self.btn_calc_sel.clicked.connect(lambda: self._compute(True))
        crow.addWidget(self.btn_calc_full)
        crow.addWidget(self.btn_calc_sel)
        sform.addLayout(crow)
        self.btn_export = QtWidgets.QPushButton("Export results to CSV…")
        self.btn_export.clicked.connect(self._export_csv)
        sform.addWidget(self.btn_export)
        v.addWidget(stab_box)

        v.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        return scroll

    def _build_stability_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.result_table = QtWidgets.QTableWidget()
        self.result_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        panel.addWidget(self.result_table)

        self.adev_plot = pg.PlotWidget()
        self.adev_plot.setBackground("w")
        self.adev_plot.setLogMode(x=True, y=True)
        self.adev_plot.showGrid(x=True, y=True, alpha=0.3)
        self.adev_plot.setLabel("bottom", "Averaging time τ", units="s")
        self.adev_plot.setLabel("left", "Allan deviation σ(τ)")
        self.adev_plot.addLegend()
        panel.addWidget(self.adev_plot)
        panel.setStretchFactor(0, 1)
        panel.setStretchFactor(1, 1)
        return panel

    # ---------------------------------------------------------------- files
    def _load_files(self, append: bool):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select FXE file(s)", "",
            "Data files (*.txt *.dat *.csv *.tsv *.asc);;All files (*)",
        )
        if not paths:
            return
        try:
            if append and self.mapping is not None:
                self.paths.extend(paths)
            else:
                self.paths = list(paths)
                # Establish the column mapping from the first file.
                first_raw = loader.parse_file(self.paths[0])
                dlg = ImportDialog(first_raw, self)
                if dlg.exec_() != QtWidgets.QDialog.Accepted:
                    return
                self.mapping = dlg.get_mapping()
            self._build_dataset()
        except Exception as exc:  # surface parsing errors to the user
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))

    def _clear_files(self):
        self.paths = []
        self.mapping = None
        self.dataset = None
        self.file_list.clear()
        self._populate_channel_checks([])
        self.plot_grid.show_channels(None, [])
        self.result_table.clear()
        self.adev_plot.clear()
        self.statusBar().showMessage("Cleared.")

    def _build_dataset(self):
        assert self.mapping is not None
        self.dataset = loader.load_files(self.paths, self.mapping, join=True)
        self.file_list.clear()
        for p in self.paths:
            self.file_list.addItem(os.path.basename(p))
        self._populate_channel_checks(self.dataset.channel_names)
        self._rebuild_plots()

        msg = (
            f"Loaded {len(self.paths)} file(s), "
            f"{self.dataset.time.size:,} samples, "
            f"gate τ₀ ≈ {self.dataset.tau0*1e3:.4g} ms, "
            f"duration {self.dataset.duration:.3g} s."
        )
        if self.dataset.gap_indices:
            msg += (f"  ⚠ {len(self.dataset.gap_indices)} timing gap(s) detected — "
                    "Allan deviation assumes uniform sampling.")
        self.statusBar().showMessage(msg)

    # -------------------------------------------------------------- channels
    def _populate_channel_checks(self, names: List[str]):
        # Clear existing.
        while self.chan_layout.count():
            item = self.chan_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.channel_checks = {}
        for i, name in enumerate(names):
            chk = QtWidgets.QCheckBox(name)
            chk.setChecked(True)
            color = _COLORS[i % len(_COLORS)]
            chk.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")
            chk.toggled.connect(self._rebuild_plots)
            self.chan_layout.addWidget(chk)
            self.channel_checks[name] = chk

    def _selected_channels(self) -> List[str]:
        return [n for n, c in self.channel_checks.items() if c.isChecked()]

    # ---------------------------------------------------------------- plots
    def _rebuild_plots(self):
        if self.dataset is None:
            return
        active = self._selected_channels()[:8]
        layout = _LAYOUTS[self.layout_combo.currentText()]
        self.plot_grid.show_channels(self.dataset, active, layout)
        self.plot_grid.set_link_x(self.link_check.isChecked())
        self.plot_grid.set_region_enabled(self.region_check.isChecked())

    def _on_region_changed(self, t0: float, t1: float):
        self.region_label.setText(f"selection: {t0:.4g} s … {t1:.4g} s  (Δ {t1-t0:.4g} s)")

    # ----------------------------------------------------------- stability
    def _toggle_ref(self, use_mean: bool):
        self.ref_spin.setEnabled(not use_mean)

    def _selected_taus(self) -> List[float]:
        taus = [val for val, chk in self.tau_checks.items() if chk.isChecked()]
        text = self.custom_taus.text().strip()
        if text:
            for tok in text.replace(";", ",").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    taus.append(float(tok))
                except ValueError:
                    pass
        return sorted(set(taus))

    def _compute(self, use_selection: bool):
        if self.dataset is None:
            QtWidgets.QMessageBox.information(self, "No data", "Load a file first.")
            return
        taus = self._selected_taus()
        if not taus:
            QtWidgets.QMessageBox.information(self, "No τ", "Select at least one averaging time.")
            return

        active = self._selected_channels()
        tau0 = self.dataset.tau0
        f_ref_override = None if self.use_mean_ref.isChecked() else float(self.ref_spin.value())

        results: Dict[str, list] = {}
        for name in active:
            freq = self.dataset.channels[name]
            if use_selection:
                mask = self.plot_grid.selection_mask(name)
                if mask is not None:
                    freq = freq[mask]
            results[name] = stability.compute_stability(
                freq, tau0, taus=taus, f_ref=f_ref_override
            )
        self.last_results = results
        self._refresh_results_view()

        span = self.plot_grid.region()
        where = (f"selection ({span[0]:.4g}–{span[1]:.4g} s)"
                 if (use_selection and span) else "full range")
        self.statusBar().showMessage(f"Computed stability over {where} for {len(active)} channel(s).")

    def _refresh_results_view(self):
        self.result_mode = self.result_mode_combo.currentData()
        self._populate_results_table(self.last_results)
        self._plot_adev(self.last_results)

    def _populate_results_table(self, results: Dict[str, list]):
        self.result_table.clear()
        if not results:
            return
        # Union of taus (by tau_used) across channels, sorted.
        tau_set = sorted({round(p.tau_used, 12) for pts in results.values() for p in pts})
        channels = list(results.keys())
        frac = (self.result_mode == "fractional")

        self.result_table.setRowCount(len(tau_set))
        self.result_table.setColumnCount(len(channels))
        self.result_table.setHorizontalHeaderLabels(channels)
        self.result_table.setVerticalHeaderLabels([self._fmt_tau(t) for t in tau_set])

        for col, name in enumerate(channels):
            by_tau = {round(p.tau_used, 12): p for p in results[name]}
            for row, tau in enumerate(tau_set):
                p = by_tau.get(tau)
                if p is None:
                    text = "—"
                else:
                    val = p.adev_frac if frac else p.adev_hz
                    err = p.err_frac if frac else p.err_hz
                    text = f"{val:.3e} ± {err:.1e}"
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(f"m={p.m}, n={p.n}" if p else "insufficient data")
                self.result_table.setItem(row, col, item)
        self.result_table.resizeColumnsToContents()
        unit = "fractional σ(τ)" if frac else "σ(τ) [Hz]"
        self.result_table.setToolTip(unit)

    def _plot_adev(self, results: Dict[str, list]):
        self.adev_plot.clear()
        frac = (self.result_mode == "fractional")
        self.adev_plot.setLabel(
            "left", "Fractional Allan deviation σ(τ)" if frac else "Allan deviation σ(τ)",
            units=None if frac else "Hz",
        )
        legend = self.adev_plot.addLegend()
        _ = legend
        for i, (name, pts) in enumerate(results.items()):
            if not pts:
                continue
            x = np.array([p.tau_used for p in pts])
            y = np.array([(p.adev_frac if frac else p.adev_hz) for p in pts])
            good = np.isfinite(y) & (y > 0)
            if not np.any(good):
                continue
            color = _COLORS[i % len(_COLORS)]
            self.adev_plot.plot(
                x[good], y[good], name=name,
                pen=pg.mkPen(color, width=2),
                symbol="o", symbolBrush=color, symbolSize=6,
            )

    @staticmethod
    def _fmt_tau(tau: float) -> str:
        if tau < 1.0:
            return f"{tau*1e3:g} ms"
        return f"{tau:g} s"

    def _export_csv(self):
        if not self.last_results:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Compute stability first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export stability CSV", "fxe_stability.csv", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["channel", "tau_s", "m", "n",
                         "adev_hz", "err_hz", "adev_fractional", "err_fractional"])
            for name, pts in self.last_results.items():
                for p in pts:
                    wr.writerow([name, p.tau_used, p.m, p.n,
                                 p.adev_hz, p.err_hz, p.adev_frac, p.err_frac])
        self.statusBar().showMessage(f"Exported results to {path}")
