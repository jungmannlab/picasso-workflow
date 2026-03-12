<#
.SYNOPSIS
    Creates a picasso-workflow GUI shortcut on the desktop.

.DESCRIPTION
    Locates the picasso-workflow-gui.exe installed by `pip install -e .`
    inside the active conda environment and writes a .lnk shortcut.

    Without -AllUsers (default): writes to your personal desktop
    (~\Desktop).  No admin rights required - use this to test first.

    With -AllUsers: writes to C:\Users\Public\Desktop so every user on
    the machine sees the shortcut.  Requires Administrator privileges.

.PARAMETER CondaEnvPath
    Full path to the conda environment that contains picasso-workflow.
    Defaults to the currently active environment ($env:CONDA_PREFIX).

    Example:
        .\deploy_gui_shortcut.ps1 -CondaEnvPath "C:\ProgramData\Anaconda3\envs\picasso-workflow"

.PARAMETER ShortcutName
    Name of the .lnk file (without extension).  Default: "picasso-workflow".

.PARAMETER AllUsers
    When specified, places the shortcut on the All-Users desktop
    (C:\Users\Public\Desktop).  Requires Administrator privileges.

.EXAMPLE
    # Test as a normal user - shortcut appears on your own desktop:
    conda activate picasso-workflow
    powershell -ExecutionPolicy Bypass -File tools\deploy_gui_shortcut.ps1

.EXAMPLE
    # Deploy to all users - run from an elevated prompt:
    powershell -ExecutionPolicy Bypass -File tools\deploy_gui_shortcut.ps1 -AllUsers
#>

param(
    [string]$CondaEnvPath = "",
    [string]$ShortcutName = "picasso-workflow",
    [switch]$AllUsers
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# 0. Guard: -AllUsers requires admin
# ---------------------------------------------------------------------------
if ($AllUsers) {
    $isAdmin = ([Security.Principal.WindowsPrincipal] `
        [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Error @"
-AllUsers requires Administrator privileges.
Re-run from an elevated PowerShell prompt, or omit -AllUsers to create
a shortcut on your own desktop for testing.
"@
        exit 1
    }
}

# ---------------------------------------------------------------------------
# 1. Resolve the conda environment
# ---------------------------------------------------------------------------
# Strategy (first match wins):
#   A. -CondaEnvPath passed explicitly
#   B. picasso-workflow-gui is on PATH (conda env is active) -> derive path
#   C. $env:CONDA_PREFIX points to the named env and contains the exe
#   D. $env:CONDA_PREFIX is the base env -> check its envs\picasso-workflow
#   E. Scan well-known installation directories

# A: explicit argument
if ($CondaEnvPath -and -not (Test-Path "$CondaEnvPath\Scripts\picasso-workflow-gui.exe")) {
    Write-Warning "Executable not found in specified path: $CondaEnvPath"
    $CondaEnvPath = ""
}

# B: exe on PATH (env is active) - most reliable when conda activate was run
if (-not $CondaEnvPath) {
    $cmd = Get-Command "picasso-workflow-gui" -ErrorAction SilentlyContinue
    if ($cmd) {
        # Scripts\ is one level below the env root
        $CondaEnvPath = Split-Path $cmd.Source -Parent | Split-Path -Parent
        Write-Host "Found via PATH: $CondaEnvPath"
    }
}

# C/D: CONDA_PREFIX (may point to the named env or to the base installation)
if (-not $CondaEnvPath -and $env:CONDA_PREFIX) {
    foreach ($candidate in @($env:CONDA_PREFIX, "$env:CONDA_PREFIX\envs\picasso-workflow")) {
        if (Test-Path "$candidate\Scripts\picasso-workflow-gui.exe") {
            $CondaEnvPath = $candidate
            Write-Host "Found via CONDA_PREFIX: $CondaEnvPath"
            break
        }
    }
}

# E: scan well-known locations (covers installs where env was not activated)
if (-not $CondaEnvPath) {
    foreach ($base in @(
        "$env:USERPROFILE\.conda",
        "$env:USERPROFILE\miniconda3",
        "$env:USERPROFILE\anaconda3",
        "$env:USERPROFILE\AppData\Local\miniconda3",
        "$env:USERPROFILE\AppData\Local\anaconda3",
        "C:\ProgramData\Miniconda3",
        "C:\ProgramData\Anaconda3"
    )) {
        $candidate = "$base\envs\picasso-workflow"
        if (Test-Path "$candidate\Scripts\picasso-workflow-gui.exe") {
            $CondaEnvPath = $candidate
            Write-Host "Found in well-known location: $CondaEnvPath"
            break
        }
    }
}

if (-not $CondaEnvPath) {
    Write-Error @"
Could not find picasso-workflow-gui.exe.
Make sure the package is installed in the active environment:
    pip install -e C:\path\to\picasso-workflow
Then re-run this script with the conda environment activated, or pass the
path explicitly:
    .\deploy_gui_shortcut.ps1 -CondaEnvPath "C:\...\envs\picasso-workflow"
"@
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Build the exe path (already verified to exist by the search above)
# ---------------------------------------------------------------------------
$ExePath = Join-Path $CondaEnvPath "Scripts\picasso-workflow-gui.exe"

# ---------------------------------------------------------------------------
# 3. Locate the icon installed with the package
# ---------------------------------------------------------------------------
$IconPath = Join-Path $CondaEnvPath "Lib\site-packages\picasso_workflow\picasso-workflow.ico"

if (-not (Test-Path $IconPath)) {
    Write-Warning "Icon not found at $IconPath - shortcut will use the default Python icon."
    $IconPath = $ExePath
}

# ---------------------------------------------------------------------------
# 4. Write the shortcut
# ---------------------------------------------------------------------------
if ($AllUsers) {
    $DesktopPath = [Environment]::GetFolderPath("CommonDesktopDirectory")
    $Scope       = "all users"
} else {
    $DesktopPath = [Environment]::GetFolderPath("Desktop")
    $Scope       = "your personal desktop"
}

$ShortcutPath = Join-Path $DesktopPath "$ShortcutName.lnk"

$Shell    = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)

$Shortcut.TargetPath       = $ExePath
$Shortcut.WorkingDirectory = $env:USERPROFILE
$Shortcut.Description      = "picasso-workflow GUI"
$Shortcut.IconLocation     = "$IconPath,0"

$Shortcut.Save()

Write-Host ""
Write-Host "Shortcut created on ${Scope}:"
Write-Host "  $ShortcutPath"
Write-Host "  -> $ExePath"
if (-not $AllUsers) {
    Write-Host ""
    Write-Host "To deploy to all users, re-run from an elevated prompt with -AllUsers:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File tools\deploy_gui_shortcut.ps1 -AllUsers"
}
