"""Dialog to map a file's columns to a timestamp and to channels.

Pre-populated from :func:`fxe_analyzer.loader.suggest_mapping`, so for a
well-formed FXE file the user usually just clicks OK.
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets

from ..loader import (
    ColumnMapping,
    DEFAULT_INVALID_VALUE,
    RawFile,
    TIME_MODES,
    suggest_mapping,
)


class ImportDialog(QtWidgets.QDialog):
    def __init__(self, raw: RawFile, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import FXE file — column mapping")
        self.resize(720, 640)
        self._raw = raw
        self._suggestion = suggest_mapping(raw)

        root = QtWidgets.QVBoxLayout(self)

        # --- file info + preview -----------------------------------------
        info = QtWidgets.QLabel(
            f"<b>{raw.path}</b><br>"
            f"detected delimiter: "
            f"<tt>{'whitespace' if raw.delimiter is None else repr(raw.delimiter)}</tt>"
            f" &nbsp; columns: <b>{raw.n_cols}</b> &nbsp; rows: <b>{raw.n_rows}</b>"
        )
        info.setWordWrap(True)
        root.addWidget(info)

        preview = QtWidgets.QPlainTextEdit()
        preview.setReadOnly(True)
        preview.setMaximumHeight(150)
        preview.setPlainText("\n".join(raw.header_lines[:6] + self._sample_rows()))
        preview.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        root.addWidget(QtWidgets.QLabel("Preview:"))
        root.addWidget(preview)

        # --- time settings ----------------------------------------------
        time_box = QtWidgets.QGroupBox("Time axis")
        form = QtWidgets.QFormLayout(time_box)

        self.time_combo = QtWidgets.QComboBox()
        self.time_combo.addItem("None — synthesise from gate time", -1)
        for c in range(raw.n_cols):
            self.time_combo.addItem(f"Column {c}", c)
        tc = self._suggestion.time_col if self._suggestion.time_col is not None else -1
        self.time_combo.setCurrentIndex(self.time_combo.findData(tc))
        form.addRow("Timestamp column:", self.time_combo)

        self.mode_combo = QtWidgets.QComboBox()
        for m in TIME_MODES:
            self.mode_combo.addItem(m, m)
        self.mode_combo.setCurrentIndex(
            max(0, self.mode_combo.findData(self._suggestion.time_mode))
        )
        form.addRow("Timestamp interpretation:", self.mode_combo)

        # Date column (only used for the 'datetime' = YYMMDD + HHMMSS mode).
        self.date_combo = QtWidgets.QComboBox()
        self.date_combo.addItem("None", -1)
        for c in range(raw.n_cols):
            self.date_combo.addItem(f"Column {c}", c)
        dc = self._suggestion.date_col if self._suggestion.date_col is not None else -1
        self.date_combo.setCurrentIndex(self.date_combo.findData(dc))
        form.addRow("Date column (YYMMDD):", self.date_combo)

        self.gate_spin = QtWidgets.QDoubleSpinBox()
        self.gate_spin.setDecimals(9)
        self.gate_spin.setRange(1e-9, 1e6)
        self.gate_spin.setValue(self._suggestion.gate_time)
        self.gate_spin.setSuffix(" s")
        form.addRow("Gate time (sample interval):", self.gate_spin)

        hint = QtWidgets.QLabel(
            "<i>index</i> = sample counter (×gate time); "
            "<i>seconds</i>/<i>unix</i> = absolute seconds; "
            "<i>mjd</i> = Modified Julian Date (days); "
            "<i>datetime</i> = date column (YYMMDD) + time column (HHMMSS.sss); "
            "<i>hhmmss</i> = time-of-day column only. Gate time is used when "
            "there is no timestamp column, or for <i>index</i> mode."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        form.addRow(hint)

        # Invalid / over-range value handling.
        self.invalid_check = QtWidgets.QCheckBox("Blank over-range / unlocked value")
        self.invalid_check.setChecked(self._suggestion.invalid_value is not None)
        self.invalid_spin = QtWidgets.QDoubleSpinBox()
        self.invalid_spin.setDecimals(3)
        self.invalid_spin.setRange(0.0, 1e18)
        self.invalid_spin.setValue(self._suggestion.invalid_value or DEFAULT_INVALID_VALUE)
        self.invalid_check.toggled.connect(self.invalid_spin.setEnabled)
        inv_row = QtWidgets.QHBoxLayout()
        inv_row.addWidget(self.invalid_check)
        inv_row.addWidget(self.invalid_spin)
        form.addRow("Invalid samples:", self._wrap(inv_row))
        root.addWidget(time_box)

        # --- channel assignment -----------------------------------------
        chan_box = QtWidgets.QGroupBox("Channels (map up to 8 columns)")
        cv = QtWidgets.QVBoxLayout(chan_box)
        self.table = QtWidgets.QTableWidget(raw.n_cols, 3)
        self.table.setHorizontalHeaderLabels(["Column", "Use as channel", "Name"])
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Stretch
        )
        self._checks = []
        self._names = []
        suggested_cols = {v: k for k, v in self._suggestion.channel_cols.items()}
        for c in range(raw.n_cols):
            self.table.setItem(c, 0, self._readonly_item(f"Column {c}"))
            chk = QtWidgets.QCheckBox()
            chk.setChecked(c in suggested_cols)
            chk.stateChanged.connect(self._enforce_limit)
            wrap = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(wrap)
            hl.addWidget(chk)
            hl.setAlignment(QtCore.Qt.AlignCenter)
            hl.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(c, 1, wrap)
            name_edit = QtWidgets.QLineEdit(suggested_cols.get(c, f"Ch{c+1}"))
            self.table.setCellWidget(c, 2, name_edit)
            self._checks.append(chk)
            self._names.append(name_edit)
        cv.addWidget(self.table)
        root.addWidget(chan_box)

        # --- buttons -----------------------------------------------------
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.time_combo.currentIndexChanged.connect(self._sync_time_channel)
        self.date_combo.currentIndexChanged.connect(self._sync_time_channel)
        self.mode_combo.currentIndexChanged.connect(self._sync_mode)
        self._sync_mode()
        self._sync_time_channel()

    # ------------------------------------------------------------- helpers
    def _sample_rows(self, n: int = 8):
        rows = []
        for r in range(min(n, self._raw.n_rows)):
            rows.append("  ".join(f"{v:g}" for v in self._raw.data[r]))
        return rows

    @staticmethod
    def _readonly_item(text: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        return item

    @staticmethod
    def _wrap(layout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _sync_mode(self):
        """Enable the date column only for 'datetime' mode."""
        is_datetime = (self.mode_combo.currentData() == "datetime")
        self.date_combo.setEnabled(is_datetime)
        self._sync_time_channel()

    def _sync_time_channel(self):
        """Disable channel check-boxes for the time and date columns."""
        tc = self.time_combo.currentData()
        dc = self.date_combo.currentData() if self.date_combo.isEnabled() else -1
        for c, chk in enumerate(self._checks):
            reserved = (c == tc) or (c == dc)
            chk.setEnabled(not reserved)
            if reserved:
                chk.setChecked(False)

    def _enforce_limit(self):
        """Keep at most 8 channels selected."""
        checked = [c for c in self._checks if c.isChecked()]
        if len(checked) > 8:
            # Uncheck the sender (the most recently toggled) if over the cap.
            sender = self.sender()
            if isinstance(sender, QtWidgets.QCheckBox):
                sender.blockSignals(True)
                sender.setChecked(False)
                sender.blockSignals(False)
                QtWidgets.QMessageBox.information(
                    self, "Limit", "You can display at most 8 channels."
                )

    def _on_accept(self):
        mapping = self.get_mapping()
        try:
            mapping.validate(self._raw.n_cols)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid mapping", str(exc))
            return
        self.accept()

    # -------------------------------------------------------------- result
    def get_mapping(self) -> ColumnMapping:
        tc = self.time_combo.currentData()
        time_col = None if tc == -1 else int(tc)
        dc = self.date_combo.currentData()
        date_col = None if dc == -1 else int(dc)
        channel_cols = {}
        for c, (chk, edit) in enumerate(zip(self._checks, self._names)):
            if chk.isChecked() and chk.isEnabled():
                name = edit.text().strip() or f"Ch{c+1}"
                channel_cols[name] = c
        return ColumnMapping(
            channel_cols=channel_cols,
            time_col=time_col,
            time_mode=self.mode_combo.currentData(),
            gate_time=float(self.gate_spin.value()),
            date_col=date_col,
            invalid_value=(float(self.invalid_spin.value())
                           if self.invalid_check.isChecked() else None),
        )
