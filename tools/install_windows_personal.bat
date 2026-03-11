@echo off
setlocal EnableDelayedExpansion

echo =====================================================
echo  picasso-workflow  --  personal install (no admin)
echo =====================================================
echo.
echo This installs picasso-workflow into a conda environment
echo and creates a shortcut on YOUR desktop for testing.
echo No administrator rights are required.
echo.

:: ---------------------------------------------------------------------------
:: 1. Find conda
:: ---------------------------------------------------------------------------
set CONDA_BASE=

:: Try conda on PATH first
where conda >nul 2>&1
if not errorlevel 1 (
    for /f "usebackq delims=" %%i in (`conda info --base 2^>nul`) do set CONDA_BASE=%%i
)

:: Fall back to common user-level install locations
if not defined CONDA_BASE (
    for %%P in (
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

:: Fall back to system-level installs
if not defined CONDA_BASE (
    for %%P in (
        "C:\ProgramData\Miniconda3"
        "C:\ProgramData\Anaconda3"
    ) do (
        if not defined CONDA_BASE (
            if exist "%%~P\Scripts\activate.bat" set CONDA_BASE=%%~P
        )
    )
)

if not defined CONDA_BASE (
    echo ERROR: conda not found.
    echo.
    echo Install Miniconda from:
    echo   https://docs.conda.io/en/latest/miniconda.html
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
:: 5. Create the desktop shortcut (personal, no admin needed)
:: ---------------------------------------------------------------------------
echo.
echo Creating shortcut on your desktop ...
powershell -ExecutionPolicy Bypass -File "%~dp0deploy_gui_shortcut.ps1" ^
    -CondaEnvPath "%CONDA_BASE%\envs\picasso-workflow"
if errorlevel 1 goto :fail

echo.
echo =====================================================
echo  Done!
echo  Double-click the "picasso-workflow" shortcut on
echo  your desktop to launch the GUI.
echo.
echo  When ready to deploy to all users, run:
echo    install_windows_allusers.bat  (as Administrator)
echo =====================================================
goto :end

:fail
echo.
echo Installation failed. See the errors above.

:end
echo.
pause
endlocal
