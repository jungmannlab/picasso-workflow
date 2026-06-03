"""Entry point wrapper for the picasso-workflow GUI.

Using pythonw / a windowless executable means that any uncaught exception
before the Qt event loop starts is completely invisible — no terminal output,
no dialog.  This module wraps the real startup so that:

  1. A crash log is always written to ~/picasso-workflow-crash.log.
  2. A Qt error dialog is shown if Qt can be imported at all.

The gui-scripts entry point in pyproject.toml points here instead of
directly to picasso_workflow.gui:main.
"""


def _hide_console_window() -> None:
    """Hide the console window on Windows so no black terminal flashes up.

    console_scripts creates an exe backed by python.exe (always present in
    conda envs), whereas gui_scripts uses pythonw.exe which may be absent.
    We therefore use console_scripts and hide the window ourselves.
    """
    import sys

    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass


def main() -> None:
    _hide_console_window()

    import sys
    from pathlib import Path

    crash_log = Path.home() / "picasso-workflow-crash.log"

    try:
        from picasso_workflow.gui import main as _gui_main

        _gui_main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        msg = traceback.format_exc()

        # Always write to a location that is writable regardless of cwd.
        try:
            crash_log.write_text(msg, encoding="utf-8")
        except Exception:
            pass

        # Best-effort Qt dialog so the user sees something on screen.
        try:
            from PyQt6 import QtWidgets

            _app = QtWidgets.QApplication.instance()
            if _app is None:
                _app = QtWidgets.QApplication(sys.argv)
            QtWidgets.QMessageBox.critical(
                None,
                "picasso-workflow failed to start",
                f"{msg}\n\nFull crash log:\n{crash_log}",
            )
        except Exception:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
