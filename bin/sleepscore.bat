@echo off
REM Windows launcher: start the sleep-scoring GUI from any folder without
REM activating conda first (counterpart of the macOS/Linux `sleepscore` script).
REM
REM The conda env name differs between PCs, so nothing is hardcoded: this finds
REM the environment where `pip install -e .` was run by looking for its
REM Scripts\sleepscore.exe entry point, and checks the install actually imports
REM (a leftover install pointing at a moved repo folder is skipped).
REM
REM Search order:
REM   1. %SLEEPSCORE_CONDA_ENV%   (set it to force a specific env name)
REM   2. the currently active env (%CONDA_PREFIX%)
REM   3. every env of every conda/mamba install found on this machine
REM
REM   sleepscore --which    print which env's command would run, don't launch
setlocal

set ROOTS="%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%USERPROFILE%\.conda" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "C:\ProgramData\miniconda3" "C:\ProgramData\Anaconda3"
set "CAND="

if defined SLEEPSCORE_CONDA_ENV (
    for %%R in (%ROOTS%) do if not defined CAND call :check "%%~R\envs\%SLEEPSCORE_CONDA_ENV%"
    if defined CAND goto :launch
    echo sleepscore: no working 'sleepscore' command in conda env "%SLEEPSCORE_CONDA_ENV%" — activate it and run: pip install -e . 1>&2
    exit /b 1
)

if defined CONDA_PREFIX call :check "%CONDA_PREFIX%"
if defined CAND goto :launch

for %%R in (%ROOTS%) do (
    if not defined CAND (
        for /d %%E in ("%%~R\envs\*") do if not defined CAND call :check "%%E"
        if not defined CAND call :check "%%~R"
    )
)
if defined CAND goto :launch

echo sleepscore: no conda env with a working install found. 1>&2
echo   Fix: conda activate ^<your-env^>, then from the repo folder run: pip install -e . 1>&2
exit /b 1

:check
REM Accept an env only if its entry point exists AND imports (skips installs
REM left behind after the repo folder was moved or deleted).
if exist "%~1\Scripts\sleepscore.exe" (
    "%~1\python.exe" -c "import sleepscore" >nul 2>&1 && set "CAND=%~1\Scripts\sleepscore.exe"
)
exit /b 0

:launch
if "%~1"=="--which" (
    echo %CAND%
    exit /b 0
)
"%CAND%" %*
exit /b %errorlevel%
