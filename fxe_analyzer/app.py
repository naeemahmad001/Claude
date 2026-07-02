"""Application entry point.

Run with either::

    python -m fxe_analyzer
    python run.py
"""

from __future__ import annotations

import sys


def main() -> int:
    # Imported lazily so the pure loader/stability modules can be used
    # (and tested) without a Qt display being available.
    from PyQt5 import QtWidgets

    from .gui.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FXE Frequency Counter Analyzer")
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
