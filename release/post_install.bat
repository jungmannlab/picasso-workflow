@echo off
rem Windows post-install: install picasso_workflow (and its PyPI deps) into the
rem freshly-laid-down conda env, then create per-user Start Menu + Desktop
rem shortcuts. constructor sets %PREFIX% to the install location.
setlocal

set "PYTHON=%PREFIX%\python.exe"

rem Bundled wheel is copied flat into %PREFIX% (via `extra_files`). --find-links
rem lets pip install it while pulling picassosr + the rest from PyPI wheels.
rem PYTHONNOUSERSITE=1 keeps a stray %APPDATA%\Python user-site from shadowing
rem the bundled env during the install (the documented ~/.local trap).
set "PYTHONNOUSERSITE=1"
"%PYTHON%" -m pip install --no-input --no-warn-script-location --find-links "%PREFIX%" picasso_workflow
if errorlevel 1 exit /b 1

rem The pip install above creates picasso-workflow-gui.exe in the env Scripts
rem dir (a console_scripts shim; _launcher.py hides its console).
set "GUI_EXE=%PREFIX%\Scripts\picasso-workflow-gui.exe"

rem A VBScript launcher sets PYTHONNOUSERSITE=1 for the running GUI (a .lnk
rem cannot set env vars) and starts the exe hidden, so there is no console
rem flash and the app never picks up user-site packages. Written line by line.
set "VBS=%PREFIX%\picasso-workflow-launcher.vbs"
> "%VBS%" echo Set sh = CreateObject("WScript.Shell")
>> "%VBS%" echo sh.Environment("PROCESS").Item("PYTHONNOUSERSITE") = "1"
>> "%VBS%" echo sh.Run """%GUI_EXE%""", 0, False

rem Per-user Start Menu + Desktop shortcuts (HKCU / user profile only -- no
rem admin, no collision between users) pointing at the VBScript launcher via
rem wscript. The icon ships as package-data, so it lands in site-packages.
set "ICON=%PREFIX%\Lib\site-packages\picasso_workflow\picasso-workflow.ico"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$startMenu = [Environment]::GetFolderPath('Programs');" ^
  "$desktop = [Environment]::GetFolderPath('Desktop');" ^
  "foreach ($dir in @($startMenu, $desktop)) {" ^
  "  $lnk = $ws.CreateShortcut((Join-Path $dir 'picasso-workflow.lnk'));" ^
  "  $lnk.TargetPath = 'wscript.exe';" ^
  "  $lnk.Arguments = '\"%VBS%\"';" ^
  "  $lnk.WorkingDirectory = '%PREFIX%';" ^
  "  if (Test-Path '%ICON%') { $lnk.IconLocation = '%ICON%' }" ^
  "  $lnk.Save();" ^
  "}"

endlocal
exit /b 0
