@echo off
rem Windows post-install: install picasso_workflow (and its PyPI deps) into the
rem freshly-laid-down conda env, then create Start Menu + Desktop shortcuts
rem (all-users or per-user, matching the install scope -- see below).
rem constructor sets %PREFIX% to the install location.
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

rem Start Menu + Desktop shortcuts pointing at the VBScript launcher via
rem wscript. An all-users install (%PREFIX% under %ALLUSERSPROFILE%, e.g.
rem C:\ProgramData -- chosen by an elevated installer) writes the COMMON
rem (all-users) folders so every account sees the shortcuts; a per-user
rem install writes that user's own folders. The icon ships as package-data,
rem so it lands in site-packages.
set "ICON=%PREFIX%\Lib\site-packages\picasso_workflow\picasso-workflow.ico"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$allUsers = '%PREFIX%'.StartsWith($env:ALLUSERSPROFILE, [System.StringComparison]::OrdinalIgnoreCase);" ^
  "$menuName = if ($allUsers) { 'CommonPrograms' } else { 'Programs' };" ^
  "$deskName = if ($allUsers) { 'CommonDesktopDirectory' } else { 'Desktop' };" ^
  "$startMenu = [Environment]::GetFolderPath($menuName);" ^
  "$desktop = [Environment]::GetFolderPath($deskName);" ^
  "foreach ($dir in @($startMenu, $desktop)) {" ^
  "  $lnk = $ws.CreateShortcut((Join-Path $dir 'picasso-workflow.lnk'));" ^
  "  $lnk.TargetPath = 'wscript.exe';" ^
  "  $lnk.Arguments = '\"%VBS%\"';" ^
  "  $lnk.WorkingDirectory = '%PREFIX%';" ^
  "  if (Test-Path '%ICON%') { $lnk.IconLocation = '%ICON%,0' }" ^
  "  $lnk.Save();" ^
  "}"

endlocal
exit /b 0
