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
# Search every plausible location and accept the first one that actually
# contains the installed exe.  This handles three common situations:
#   a. -CondaEnvPath was passed explicitly
#   b. $env:CONDA_PREFIX points to the named env (conda was activated)
#   c. $env:CONDA_PREFIX points to the base installation (base env active)
#   d. conda is not on PATH at all in this PowerShell session

$candidates = [System.Collections.Generic.List[string]]::new()

# Explicit argument takes priority
if ($CondaEnvPath) { $candidates.Add($CondaEnvPath) }

# CONDA_PREFIX: try both the value itself and its envs\picasso-workflow child
if ($env:CONDA_PREFIX) {
    $candidates.Add($env:CONDA_PREFIX)
    $candidates.Add("$env:CONDA_PREFIX\envs\picasso-workflow")
}

# Well-known installation directories
foreach ($base in @(
    "$env:USERPROFILE\miniconda3",
    "$env:USERPROFILE\anaconda3",
    "$env:USERPROFILE\AppData\Local\miniconda3",
    "$env:USERPROFILE\AppData\Local\anaconda3",
    "C:\ProgramData\Miniconda3",
    "C:\ProgramData\Anaconda3"
)) {
    $candidates.Add("$base\envs\picasso-workflow")
}

$CondaEnvPath = ""
foreach ($c in $candidates) {
    if (Test-Path "$c\Scripts\picasso-workflow-gui.exe") {
        $CondaEnvPath = $c
        Write-Host "Found conda environment: $CondaEnvPath"
        break
    }
}

if (-not $CondaEnvPath) {
    Write-Error @"
Could not find picasso-workflow-gui.exe in any known location.
Make sure the package is installed, then pass the env path explicitly:
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
