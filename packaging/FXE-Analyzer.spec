# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the FXE Frequency Counter Analyzer.

Builds a single-file, windowed executable.  Invoke from the repository
root (build_windows.bat does this for you):

    pyinstaller --noconfirm --clean packaging/FXE-Analyzer.spec

pyqtgraph performs several dynamic imports (its exporters and graphics
items), so we pull in all of its submodules and data files explicitly.
"""

import os

# The spec is executed with the repo root as the working directory.
ROOT = os.path.abspath(os.getcwd())

# Our code imports pyqtgraph, PyQt5 and the exporters with ordinary
# `import` statements, so PyInstaller's static analysis already follows
# them.  We only nudge it toward pyqtgraph's exporter package (used for
# SVG/PNG export) to be safe, and deliberately avoid collecting all of
# pyqtgraph (its `examples` subpackage crashes headless collectors).
hiddenimports = ["pyqtgraph.exporters"]
datas = []

a = Analysis(
    [os.path.join(ROOT, "run.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Keep the binary lean: these are not used by the app.
    excludes=["matplotlib", "pandas", "pytest", "tkinter", "PySide2", "PySide6"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FXE-Analyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # windowed GUI app (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
