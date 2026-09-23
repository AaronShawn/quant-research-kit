@echo off
setlocal enabledelayedexpansion
REM ===============================================================
REM  Quant Research Kit - one-click environment setup (Windows)
REM
REM  Creates a .venv inside the project and installs the full
REM  VeighNa + Qlib + akshare stack.
REM
REM  Verified on Python 3.11. Python 3.13 may fail on vnpy_riskmanager
REM  (see the note in requirements.txt).
REM ===============================================================

set "HERE=%~dp0"
pushd "%HERE%.."
set "PROJ=%CD%"
popd
set "VENV=%PROJ%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"

echo.
echo ================================================================
echo   Quant Research Kit - environment setup
echo ================================================================
echo   project : %PROJ%
echo   venv    : %VENV%
echo.

REM ---- 1. locate a suitable interpreter (3.10 - 3.12) ----
set "SYSPY="
for %%V in (3.12 3.11 3.10) do (
    if not defined SYSPY (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 set "SYSPY=py -%%V"
    )
)
if not defined SYSPY (
    python -c "import sys; assert (3,10)<=sys.version_info<(3,13)" >nul 2>&1
    if not errorlevel 1 set "SYSPY=python"
)
if not defined SYSPY (
    echo [ERROR] No suitable Python 3.10-3.12 found.
    echo         Install one from https://www.python.org/downloads/
    echo         ^(Python 3.13+ is not supported: vnpy_riskmanager has no wheel^)
    pause
    exit /b 1
)
echo [1/4] interpreter: %SYSPY%
%SYSPY% -c "import sys; print('      version    :', sys.version.split()[0])"

REM ---- 2. create venv ----
if exist "%PY%" (
    echo [2/4] venv already exists, reusing it
) else (
    echo [2/4] creating venv ...
    %SYSPY% -m venv "%VENV%"
    if errorlevel 1 goto :fail
)

echo [3/4] upgrading pip ...
"%PY%" -m pip install -q -U pip -i %MIRROR%
if errorlevel 1 goto :fail

echo [4/4] installing packages (this downloads ~400MB, be patient) ...
"%PY%" -m pip install -i %MIRROR% -r "%PROJ%\setup\requirements.txt"
if errorlevel 1 goto :fail

echo.
echo ================================================================
echo   Done.  Next step:
echo.
echo     .venv\Scripts\python.exe scripts\00_check_env.py
echo.
echo   Expect "全部 15 项通过". If something is missing, see the
echo   quant-env-inventory skill - the package may live in another
echo   interpreter on this machine.
echo ================================================================
pause
exit /b 0

:fail
echo.
echo [FAILED] See the error above.
echo   If it is a build error on vnpy_riskmanager, pin it to 1.1.0
echo   (already done in requirements.txt) or use Python 3.11.
pause
exit /b 1
