# Create the picasso-workflow Start Menu + Desktop shortcuts.
#
# A standalone script invoked by post_install.bat (`powershell -File`), so the
# shortcut logic lives in real PowerShell instead of an inline one-liner embedded
# in the .bat -- the cmd<->PowerShell quote-escaping of the inline form silently
# failed to create any shortcut on some installs. Shortcuts are best-effort: a
# failure here is logged and swallowed (exit 0) so it never aborts an otherwise
# working install (the app still runs via its VBScript launcher).
#
# The shortcut targets `wscript <prefix>\picasso-workflow-launcher.vbs` (written
# by post_install.bat), which sets PYTHONNOUSERSITE=1 and starts the GUI hidden.
param(
    [string]$Prefix = $env:PREFIX
)

$log = Join-Path $Prefix "shortcut-install.log"
function Write-Log($message) {
    try {
        Add-Content -Path $log -Value (
            (Get-Date).ToString("s") + "  " + $message
        )
    } catch {}
}

try {
    $vbs = Join-Path $Prefix "picasso-workflow-launcher.vbs"
    $icon = Join-Path $Prefix `
        "Lib\site-packages\picasso_workflow\picasso-workflow.ico"

    # An all-users install lives under %ALLUSERSPROFILE% (e.g. C:\ProgramData);
    # write the COMMON (all-users) menu/desktop so every account sees it. A
    # per-user install writes that user's own folders.
    $allUsers = $Prefix.StartsWith(
        $env:ALLUSERSPROFILE, [System.StringComparison]::OrdinalIgnoreCase
    )
    $menuName = if ($allUsers) { "CommonPrograms" } else { "Programs" }
    $deskName = if ($allUsers) { "CommonDesktopDirectory" } else { "Desktop" }
    $dirs = @(
        [Environment]::GetFolderPath($menuName),
        [Environment]::GetFolderPath($deskName)
    )

    $ws = New-Object -ComObject WScript.Shell
    foreach ($dir in $dirs) {
        if ([string]::IsNullOrEmpty($dir)) { continue }
        $lnkPath = Join-Path $dir "picasso-workflow.lnk"
        $lnk = $ws.CreateShortcut($lnkPath)
        $lnk.TargetPath = "wscript.exe"
        # Quote the .vbs path so a space in the install dir is handled.
        $lnk.Arguments = '"' + $vbs + '"'
        $lnk.WorkingDirectory = $Prefix
        if (Test-Path $icon) { $lnk.IconLocation = "$icon,0" }
        $lnk.Save()
        Write-Log "created $lnkPath"
    }
    Write-Log "shortcuts done"
} catch {
    Write-Log ("ERROR creating shortcuts: " + $_.Exception.Message)
}
exit 0
