' Create the picasso-workflow Start Menu + Desktop shortcuts.
'
' Invoked by post_install.bat via
'   cscript //nologo create_shortcuts.vbs "<prefix>"
'
' VBScript (cscript) is used INSTEAD OF PowerShell because managed / enterprise
' Windows frequently blocks PowerShell script execution via Group Policy
' (ExecutionPolicy Restricted/AllSigned or Constrained Language Mode). A machine
' ExecutionPolicy set by GPO OVERRIDES the -ExecutionPolicy Bypass flag, so the
' old `powershell -File create_shortcuts.ps1` silently ran nothing -- no shortcut
' and no log. cscript is not subject to the PowerShell execution policy and ships
' on every Windows, so it runs where PowerShell is locked down. Shortcuts are
' best-effort: errors are logged and swallowed (exit 0) so a hiccup never fails
' an otherwise-working install (the app still runs via its VBScript launcher).
'
' The shortcut targets `wscript <prefix>\picasso-workflow-launcher.vbs` (written
' by post_install.bat), which sets PYTHONNOUSERSITE=1 and starts the GUI hidden.

Option Explicit

Dim sh, fso, prefix, logPath, allUsersProfile, isAllUsers
Dim vbsPath, iconPath, menuDir, deskDir

Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

If WScript.Arguments.Count > 0 Then
    prefix = WScript.Arguments(0)
Else
    prefix = sh.ExpandEnvironmentStrings("%PREFIX%")
End If

logPath = fso.BuildPath(prefix, "shortcut-install.log")
WriteLog "create_shortcuts started (Prefix=" & prefix & ")"

On Error Resume Next

vbsPath = fso.BuildPath(prefix, "picasso-workflow-launcher.vbs")
iconPath = fso.BuildPath( _
    prefix, "Lib\site-packages\picasso_workflow\picasso-workflow.ico")

' An all-users install lives under %ALLUSERSPROFILE% (e.g. C:\ProgramData);
' write the COMMON (all-users) menu/desktop so every account sees it. A per-user
' install writes that user's own folders.
allUsersProfile = sh.ExpandEnvironmentStrings("%ALLUSERSPROFILE%")
isAllUsers = (InStr(1, prefix, allUsersProfile, vbTextCompare) = 1)
If isAllUsers Then
    menuDir = sh.SpecialFolders("AllUsersPrograms")
    deskDir = sh.SpecialFolders("AllUsersDesktop")
Else
    menuDir = sh.SpecialFolders("Programs")
    deskDir = sh.SpecialFolders("Desktop")
End If

MakeShortcut menuDir
MakeShortcut deskDir

WriteLog "shortcuts done"
WScript.Quit 0


Sub MakeShortcut(dir)
    Dim lnkPath, lnk
    If Len(dir) = 0 Then
        WriteLog "skip: empty shortcut folder"
        Exit Sub
    End If
    lnkPath = fso.BuildPath(dir, "picasso-workflow.lnk")
    Set lnk = sh.CreateShortcut(lnkPath)
    lnk.TargetPath = "wscript.exe"
    ' Quote the .vbs path so a space in the install dir is handled.
    lnk.Arguments = """" & vbsPath & """"
    lnk.WorkingDirectory = prefix
    If fso.FileExists(iconPath) Then lnk.IconLocation = iconPath & ",0"
    lnk.Save
    If Err.Number <> 0 Then
        WriteLog "ERROR creating " & lnkPath & ": " & Err.Description
        Err.Clear
    Else
        WriteLog "created " & lnkPath
    End If
End Sub


Sub WriteLog(message)
    Dim f
    On Error Resume Next
    ' 8 = ForAppending, create if absent.
    Set f = fso.OpenTextFile(logPath, 8, True)
    f.WriteLine Now & "  " & message
    f.Close
    On Error GoTo 0
End Sub
