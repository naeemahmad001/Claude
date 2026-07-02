"""A grid of interactive channel plots with a shared time-region selector.

Uses pyqtgraph for fast, mouse-driven zoom/pan on large records.  Up to
eight channels are shown at once, one plot each.  All plots can share a
common (linked) x-axis so zooming one zooms them together, and a movable
"region" spans every plot to pick a time interval for analysis.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from ..loader import Dataset

# Distinct, colour-blind-friendly-ish series colours for up to 8 channels.
_CHANNEL_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#72B7B2", "#EECA3B", "#B279A2", "#FF9DA6",
]


def auto_layout(n: int) -> Tuple[int, int]:
    """Choose a (rows, cols) grid for ``n`` plots (n in 1..8)."""
    presets = {
        1: (1, 1), 2: (1, 2), 3: (3, 1), 4: (2, 2),
        5: (3, 2), 6: (3, 2), 7: (4, 2), 8: (4, 2),
    }
    if n in presets:
        return presets[n]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


class PlotGrid(QtWidgets.QWidget):
    """Widget holding the channel plots and the shared region selector."""

    # Emitted with (t0, t1) in seconds-relative-to-record-start.
    regionChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOptions(antialias=False, background="w", foreground="k")

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        self._layout.addWidget(self.glw)

        self._dataset: Optional[Dataset] = None
        self._t0_abs: float = 0.0
        self._plots: Dict[str, pg.PlotItem] = {}
        self._regions: List[pg.LinearRegionItem] = []
        self._link_x = True
        self._region_enabled = True
        self._region_span: Optional[Tuple[float, float]] = None
        self._suppress = False  # reentrancy guard for region syncing

    # ------------------------------------------------------------------ API
    def set_link_x(self, enabled: bool) -> None:
        self._link_x = enabled
        self._relink_x()

    def set_region_enabled(self, enabled: bool) -> None:
        self._region_enabled = enabled
        for r in self._regions:
            r.setVisible(enabled)
        if enabled and self._region_span is None and self._dataset is not None:
            self.reset_region()

    def show_channels(
        self,
        dataset: Dataset,
        active: List[str],
        layout: Optional[Tuple[int, int]] = None,
    ) -> None:
        """(Re)build the grid to display the given channels."""
        self._dataset = dataset
        self.glw.clear()
        self._plots.clear()
        self._regions.clear()

        if dataset is None or not active:
            return

        self._t0_abs = float(dataset.time[0]) if dataset.time.size else 0.0
        x = dataset.time - self._t0_abs  # relative seconds, small numbers

        rows, cols = layout or auto_layout(len(active))
        first_plot: Optional[pg.PlotItem] = None

        for i, name in enumerate(active):
            r, c = divmod(i, cols)
            plot: pg.PlotItem = self.glw.addPlot(row=r, col=c)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("bottom", "Time", units="s")
            plot.setLabel("left", f"{name}", units="Hz")
            plot.setTitle(name)
            plot.addLegend(offset=(10, 5))

            y = dataset.channels[name]
            color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
            # autoDownsample keeps very large records responsive by drawing
            # at most ~one point per screen pixel.  (clipToView is avoided
            # because it interacts badly with linked x-axes in pyqtgraph.)
            plot.plot(
                x, y, pen=pg.mkPen(color, width=1), name=name,
                autoDownsample=True,
            )
            plot.enableAutoRange()

            # A region item on every plot; kept in sync across the grid.
            region = pg.LinearRegionItem(brush=(100, 100, 200, 40))
            region.setZValue(-10)
            plot.addItem(region)
            region.sigRegionChanged.connect(self._on_region_changed)
            region.setVisible(self._region_enabled)
            self._regions.append(region)

            self._plots[name] = plot
            if first_plot is None:
                first_plot = plot

        self._relink_x()
        self.reset_region()

    def reset_region(self) -> None:
        """Reset the selector to span the whole record."""
        if self._dataset is None or self._dataset.time.size == 0:
            return
        span = (
            0.0,
            float(self._dataset.time[-1] - self._t0_abs),
        )
        self.set_region(*span)

    def set_region(self, t0: float, t1: float) -> None:
        self._region_span = (t0, t1)
        self._suppress = True
        try:
            for r in self._regions:
                r.setRegion((t0, t1))
        finally:
            self._suppress = False
        self.regionChanged.emit(t0, t1)

    def region(self) -> Optional[Tuple[float, float]]:
        """Current selection in seconds relative to the record start."""
        if not self._region_enabled or self._region_span is None:
            return None
        return self._region_span

    def selection_mask(self, name: str) -> Optional[np.ndarray]:
        """Boolean mask into a channel for the current time selection."""
        if self._dataset is None:
            return None
        span = self.region()
        if span is None:
            return None
        x = self._dataset.time - self._t0_abs
        return (x >= span[0]) & (x <= span[1])

    # -------------------------------------------------------------- internals
    def _relink_x(self) -> None:
        plots = list(self._plots.values())
        if not plots:
            return
        for p in plots[1:]:
            p.setXLink(plots[0] if self._link_x else None)

    def _on_region_changed(self, sender) -> None:
        if self._suppress:
            return
        t0, t1 = sender.getRegion()
        self._region_span = (float(t0), float(t1))
        self._suppress = True
        try:
            for r in self._regions:
                if r is not sender:
                    r.setRegion((t0, t1))
        finally:
            self._suppress = False
        self.regionChanged.emit(float(t0), float(t1))
