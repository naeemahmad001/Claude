@echo off
REM One-command launcher for Windows.
REM Creates a local virtual environment, installs dependencies (first run
REM only), then starts the FXE analyzer GUI. Double-click, or run in cmd.
setlocal
cd /d "%~dp0"

set "VENV=.venv"

if not exist "%VENV%\Scripts\python.exe" (
    echo Creating virtual environment in %VENV% ...
    py -3 -m venv "%VENV%" || python -m venv "%VENV%"
)

call "%VENV%\Scripts\activate.bat"

python -c "import PyQt5, pyqtgraph, numpy" 2>nul
if errorlevel 1 (
    echo Installing dependencies ...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)

python run.py %*
if errorlevel 1 pause
endlocal
