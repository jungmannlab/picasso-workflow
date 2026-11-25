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

# Must match the id gui.py main() pins via
# SetCurrentProcessExplicitAppUserModelID: Windows groups the running window's
# taskbar button under this AppUserModelID, then sources that button's icon
# from the Start-Menu/Desktop shortcut carrying the SAME id. Without it stamped
# on the shortcut, the regrouped button has no shortcut to resolve an icon from
# and shows the generic Windows logo (even though the window icon is set).
$AppUserModelID = "org.jungmannlab.picasso-workflow"

$log = Join-Path $Prefix "shortcut-install.log"
function Write-Log($message) {
    try {
        Add-Content -Path $log -Value (
            (Get-Date).ToString("s") + "  " + $message
        )
    } catch {}
}

# Compile the IPropertyStore interop once. WScript.Shell cannot set a shortcut's
# System.AppUserModel.ID property, so we load the .lnk via the shell's
# IPropertyStore and set PKEY_AppUserModel_ID ({9F4C2855-...}, 5). Best-effort:
# if compilation is unavailable the taskbar icon simply stays as-is.
$canSetAppId = $false
try {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace PWShortcut {
    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    internal class CShellLink { }
    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown),
     Guid("0000010b-0000-0000-C000-000000000046")]
    internal interface IPersistFile {
        void GetClassID(out Guid pClassID);
        [PreserveSig] int IsDirty();
        void Load([MarshalAs(UnmanagedType.LPWStr)] string n, int m);
        void Save([MarshalAs(UnmanagedType.LPWStr)] string n,
                  [MarshalAs(UnmanagedType.Bool)] bool remember);
        void SaveCompleted([MarshalAs(UnmanagedType.LPWStr)] string n);
        void GetCurFile([MarshalAs(UnmanagedType.LPWStr)] out string n);
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    internal struct PROPERTYKEY { public Guid fmtid; public uint pid; }
    [StructLayout(LayoutKind.Sequential)]
    internal class PROPVARIANT {
        public ushort vt; public ushort r1; public ushort r2; public ushort r3;
        public IntPtr p; public IntPtr p2;
    }
    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown),
     Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
    internal interface IPropertyStore {
        [PreserveSig] int GetCount(out uint c);
        [PreserveSig] int GetAt(uint i, out PROPERTYKEY k);
        [PreserveSig] int GetValue(ref PROPERTYKEY k,
                                   [In, Out] PROPVARIANT v);
        [PreserveSig] int SetValue(ref PROPERTYKEY k, [In] PROPVARIANT v);
        [PreserveSig] int Commit();
    }
    public static class ShortcutAppId {
        [DllImport("propsys.dll", CharSet = CharSet.Unicode,
                   PreserveSig = false)]
        private static extern void InitPropVariantFromString(
            [MarshalAs(UnmanagedType.LPWStr)] string psz,
            [Out] PROPVARIANT pv);
        [DllImport("ole32.dll")]
        private static extern int PropVariantClear([In, Out] PROPVARIANT pv);
        public static void Set(string lnk, string appId) {
            var link = new CShellLink();
            try {
                ((IPersistFile)link).Load(lnk, 2); // STGM_READWRITE
                var store = (IPropertyStore)link;
                var key = new PROPERTYKEY {
                    fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"),
                    pid = 5 };
                var pv = new PROPVARIANT();
                InitPropVariantFromString(appId, pv);
                try {
                    int hr = store.SetValue(ref key, pv);
                    if (hr != 0) throw new COMException("SetValue", hr);
                    hr = store.Commit();
                    if (hr != 0) throw new COMException("Commit", hr);
                } finally { PropVariantClear(pv); }
                ((IPersistFile)link).Save(lnk, true);
            } finally { Marshal.ReleaseComObject(link); }
        }
    }
}
'@
    $canSetAppId = $true
} catch {
    Write-Log ("AppUserModelID interop unavailable: " + $_.Exception.Message)
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
        # Stamp the AppUserModelID so the taskbar button resolves this
        # shortcut's icon (best-effort; the .lnk is already saved above).
        if ($canSetAppId) {
            try {
                [PWShortcut.ShortcutAppId]::Set($lnkPath, $AppUserModelID)
                Write-Log "stamped AppUserModelID on $lnkPath"
            } catch {
                Write-Log (
                    "could not stamp AppUserModelID on " + $lnkPath +
                    ": " + $_.Exception.Message
                )
            }
        }
    }
    Write-Log "shortcuts done"
} catch {
    Write-Log ("ERROR creating shortcuts: " + $_.Exception.Message)
}
exit 0
