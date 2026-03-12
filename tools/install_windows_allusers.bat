@echo off
setlocal EnableDelayedExpansion

:: ---------------------------------------------------------------------------
:: Self-elevate: if not already running as Administrator, re-launch elevated.
:: ---------------------------------------------------------------------------
net session >nul 2>&1
if errorlevel 1 (
    echo Requesting Administrator privileges ...
    powershell -Command "Start-Process cmd.exe -ArgumentList '/c ""%~f0""' -Verb RunAs -Wait"
    exit /b
)

echo =====================================================
echo  picasso-workflow  --  all-users install (admin)
echo =====================================================
echo.
echo This installs picasso-workflow into a shared conda
echo environment and creates a shortcut on the desktop
echo of EVERY user on this machine.
echo.

:: ---------------------------------------------------------------------------
:: 1. Find conda (prefer system-level installs for shared deployments)
:: ---------------------------------------------------------------------------
set CONDA_BASE=

where conda >nul 2>&1
if not errorlevel 1 (
    for /f "usebackq delims=" %%i in (`conda info --base 2^>nul`) do set CONDA_BASE=%%i
)

if not defined CONDA_BASE (
    for %%P in (
        "C:\ProgramData\Miniconda3"
        "C:\ProgramData\Anaconda3"
        "%USERPROFILE%\miniconda3"
        "%USERPROFILE%\anaconda3"
        "%USERPROFILE%\AppData\Local\miniconda3"
        "%USERPROFILE%\AppData\Local\anaconda3"
    ) do (
        if not defined CONDA_BASE (
            if exist "%%~P\Scripts\activate.bat" set CONDA_BASE=%%~P
        )
    )
)

if not defined CONDA_BASE (
    echo ERROR: conda not found.
    echo.
    echo For a shared all-users deployment, install Miniconda system-wide:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo   (choose "Install for all users" in the installer)
    echo Then re-run this script.
    goto :fail
)

echo Found conda at: %CONDA_BASE%
echo.

:: ---------------------------------------------------------------------------
:: 2. Initialise conda for this cmd session
:: ---------------------------------------------------------------------------
call "%CONDA_BASE%\Scripts\activate.bat" "%CONDA_BASE%"

:: ---------------------------------------------------------------------------
:: 3. Create or update the conda environment
:: ---------------------------------------------------------------------------
conda env list | findstr /C:"picasso-workflow" >nul 2>&1
if errorlevel 1 (
    echo Creating conda environment "picasso-workflow" with Python 3.10 ...
    call conda create -n picasso-workflow python=3.10 -y
    if errorlevel 1 goto :fail
) else (
    echo Conda environment "picasso-workflow" already exists.
)

call conda activate picasso-workflow
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment.
    goto :fail
)

:: ---------------------------------------------------------------------------
:: 4. Install picasso-workflow
:: ---------------------------------------------------------------------------
set PROJECT_DIR=%~dp0..
echo.
echo Installing picasso-workflow from: %PROJECT_DIR%
pip install -e "%PROJECT_DIR%"
if errorlevel 1 goto :fail

:: ---------------------------------------------------------------------------
:: 5. Create the All-Users desktop shortcut
:: ---------------------------------------------------------------------------
echo.
echo Creating shortcut on the All-Users desktop ...
powershell -ExecutionPolicy Bypass -File "%~dp0deploy_gui_shortcut.ps1" ^
    -CondaEnvPath "%CONDA_BASE%\envs\picasso-workflow" ^
    -AllUsers
if errorlevel 1 goto :fail

echo.
echo =====================================================
echo  Done!
echo  The "picasso-workflow" shortcut now appears on the
echo  desktop of every user on this machine.
echo =====================================================
goto :end

:fail
echo.
echo Installation failed. See the errors above.

:end
echo.
pause
endlocal
