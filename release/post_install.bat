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

rem Start Menu + Desktop shortcuts pointing at the VBScript launcher. The logic
rem lives in a standalone PowerShell script (create_shortcuts.ps1, copied flat
rem into %PREFIX% via extra_files) invoked with `-File` -- the previous inline
rem one-liner relied on fragile cmd<->PowerShell quote-escaping that silently
rem created no shortcut on some installs. Shortcuts are best-effort: the script
rem logs and swallows its own errors (and we ignore its exit code) so a shortcut
rem hiccup never fails an otherwise-working install.
powershell -NoProfile -ExecutionPolicy Bypass -File ^
  "%PREFIX%\create_shortcuts.ps1" -Prefix "%PREFIX%"

rem Mark this as a one-click-installer (bundled) install, so the GUI offers
rem in-app self-update (pip-upgrade in place) only here, never a dev env.
> "%PREFIX%\.picasso-workflow-bundled" echo bundled

endlocal
exit /b 0
