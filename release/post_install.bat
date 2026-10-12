@echo off
rem Windows post-install: install picasso_workflow (and its PyPI deps) into the
rem freshly-laid-down conda env, then create per-user Start Menu + Desktop
rem shortcuts. constructor sets %PREFIX% to the install location.
setlocal

set "PYTHON=%PREFIX%\python.exe"

rem Bundled wheel is in %PREFIX%\wheels\ (via `extra_files`). --find-links lets
rem pip install it while pulling picassosr + the rest from PyPI wheels.
"%PYTHON%" -m pip install --no-input --no-warn-script-location --find-links "%PREFIX%\wheels" picasso_workflow
if errorlevel 1 exit /b 1

rem The pip install above creates picasso-workflow-gui.exe in the env Scripts
rem dir. Hide the console window via the gui launcher (_launcher.py already
rem calls ShowWindow(SW_HIDE); the .exe is a console_scripts shim).
set "GUI_EXE=%PREFIX%\Scripts\picasso-workflow-gui.exe"

rem Per-user Start Menu + Desktop shortcuts (HKCU / user profile only -- no
rem admin, no collision between users). Built with a throwaway PowerShell step.
rem The icon ships as package-data, so it lands in site-packages.
set "ICON=%PREFIX%\Lib\site-packages\picasso_workflow\picasso-workflow.ico"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$startMenu = [Environment]::GetFolderPath('Programs');" ^
  "$desktop = [Environment]::GetFolderPath('Desktop');" ^
  "foreach ($dir in @($startMenu, $desktop)) {" ^
  "  $lnk = $ws.CreateShortcut((Join-Path $dir 'picasso-workflow.lnk'));" ^
  "  $lnk.TargetPath = '%GUI_EXE%';" ^
  "  $lnk.WorkingDirectory = '%PREFIX%';" ^
  "  if (Test-Path '%ICON%') { $lnk.IconLocation = '%ICON%' }" ^
  "  $lnk.Save();" ^
  "}"

endlocal
exit /b 0
