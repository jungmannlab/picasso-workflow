@echo off
rem Windows post-install: install picasso_workflow (and its PyPI deps) into the
rem freshly-laid-down conda env, then create Start Menu + Desktop shortcuts
rem (all-users or per-user, matching the install scope -- see below).
rem constructor sets %PREFIX% to the install location.
setlocal

set "PYTHON=%PREFIX%\python.exe"

rem Unconditional install log so a failed install is self-diagnosing even when
rem nothing opens a console (the user can send %PREFIX%\post_install.log).
set "PILOG=%PREFIX%\post_install.log"
> "%PILOG%" echo post_install.bat started  PREFIX=%PREFIX%

rem Bundled wheel is copied flat into %PREFIX% (via `extra_files`). --find-links
rem lets pip install it while pulling picassosr + the rest from PyPI wheels.
rem PYTHONNOUSERSITE=1 keeps a stray %APPDATA%\Python user-site from shadowing
rem the bundled env during the install (the documented ~/.local trap).
set "PYTHONNOUSERSITE=1"
"%PYTHON%" -m pip install --no-input --no-warn-script-location --find-links "%PREFIX%" picasso_workflow
set "PIPRC=%errorlevel%"
>> "%PILOG%" echo pip install exit=%PIPRC%
if not "%PIPRC%"=="0" exit /b %PIPRC%

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

rem Start Menu + Desktop shortcuts pointing at the VBScript launcher. Creation
rem runs via cscript (create_shortcuts.vbs, copied flat into %PREFIX% via
rem extra_files) rather than PowerShell: managed / enterprise Windows often
rem blocks PowerShell script execution by Group Policy (ExecutionPolicy /
rem Constrained Language Mode), and a machine-policy ExecutionPolicy OVERRIDES
rem the -ExecutionPolicy Bypass flag -- so `powershell -File ...` silently
rem created no shortcut and no log. cscript is not subject to that policy.
rem Shortcuts are best-effort: the script logs and swallows its own errors (and
rem we ignore its exit code) so a shortcut hiccup never fails the install.
>> "%PILOG%" echo creating shortcuts via cscript
cscript //nologo "%PREFIX%\create_shortcuts.vbs" "%PREFIX%"
>> "%PILOG%" echo cscript exit=%errorlevel%

rem Mark this as a one-click-installer (bundled) install, so the GUI offers
rem in-app self-update (pip-upgrade in place) only here, never a dev env.
> "%PREFIX%\.picasso-workflow-bundled" echo bundled

>> "%PILOG%" echo post_install.bat done
endlocal
exit /b 0
