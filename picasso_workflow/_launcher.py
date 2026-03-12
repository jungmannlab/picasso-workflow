"""Entry point wrapper for the picasso-workflow GUI.

Using pythonw / a windowless executable means that any uncaught exception
before the Qt event loop starts is completely invisible — no terminal output,
no dialog.  This module wraps the real startup so that:

  1. A crash log is always written to ~/picasso-workflow-crash.log.
  2. A Qt error dialog is shown if Qt can be imported at all.

The gui-scripts entry point in pyproject.toml points here instead of
directly to picasso_workflow.gui:main.
"""


def main() -> None:
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
            from PyQt5 import QtWidgets

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
