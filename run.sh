#!/usr/bin/env bash
# One-command launcher for macOS / Linux.
# Creates a local virtual environment, installs dependencies (first run
# only), then starts the FXE analyzer GUI.
set -e

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment in $VENV ..."
    "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Install/refresh dependencies only when needed.
if ! python -c "import PyQt5, pyqtgraph, numpy" 2>/dev/null; then
    echo "Installing dependencies ..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

exec python run.py "$@"
