@echo off
REM =====================================================================
REM  Build a standalone Windows executable (dist\FXE-Analyzer.exe).
REM  Run this ON A WINDOWS MACHINE (double-click, or run in cmd).
REM  Requires Python 3.9+ installed and on PATH.
REM  No Python is needed to RUN the resulting .exe afterwards.
REM =====================================================================
setlocal
cd /d "%~dp0"

set "VENV=.venv-build"

if not exist "%VENV%\Scripts\python.exe" (
    echo Creating build virtual environment in %VENV% ...
    py -3 -m venv "%VENV%" || python -m venv "%VENV%"
)

call "%VENV%\Scripts\activate.bat"

echo Installing build dependencies ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

echo.
echo Building FXE-Analyzer.exe (this can take a few minutes) ...
pyinstaller --noconfirm --clean packaging\FXE-Analyzer.spec

echo.
if exist "dist\FXE-Analyzer.exe" (
    echo ============================================================
    echo  Done!  Your program is:  dist\FXE-Analyzer.exe
    echo  You can copy that single file anywhere and double-click it.
    echo ============================================================
) else (
    echo Build did not produce dist\FXE-Analyzer.exe - see messages above.
)
pause
endlocal
