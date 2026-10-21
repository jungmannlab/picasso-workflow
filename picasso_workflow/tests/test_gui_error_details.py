#!/usr/bin/env python
"""GUI tests for the "Error details" pane in the Run tab.

The pane shows failed modules' tracebacks, sourced from the progress
states; for local runs it falls back to the traceback recorded in
WorkflowRunner.yaml (runs predating the progress-error propagation) and,
when the run died without any recorded module failure, to the tail of
local_run.log.

Skips where a Qt GUI cannot be constructed.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

from PyQt6 import QtWidgets  # noqa: E402

from picasso_workflow import gui  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    try:
        win = gui.Window()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not construct GUI window: {e}")
    yield win
    win.close()


def _failed_state(error=None, modules=None):
    return {
        "kind": "single",
        "report_name": "rep_123456-0000",
        "state": "failed",
        "modules": modules
        or [
            {"i": 0, "name": "load_dataset_movie", "status": "done"},
            {"i": 1, "name": "identify", "status": "failed", "error": error},
        ],
    }


def test_progress_error_shown(window):
    window._monitor_local_folder = None
    window._update_error_details(
        [_failed_state(error="AutoPicassoError: kaboom\nTraceback (...)")]
    )
    text = window.error_details_display.toPlainText()
    assert "identify" in text
    assert "kaboom" in text


def test_no_failures_clears_pane(window):
    window.error_details_display.setPlainText("stale")
    window._monitor_local_folder = None
    window._update_error_details(
        [_failed_state(modules=[{"i": 0, "name": "load", "status": "done"}])]
    )
    # state failed but no module failure and no local folder: nothing shown
    assert window.error_details_display.toPlainText() == ""


def test_yaml_traceback_fallback(window, tmp_path):
    """A failed module without a progress error (pre-existing run) gets its
    traceback from the run's WorkflowRunner.yaml."""
    run_dir = tmp_path / "rep_123456-0000"
    run_dir.mkdir()
    (run_dir / "WorkflowRunner.yaml").write_text(
        "results:\n"
        "  01_identify:\n"
        "    success: false\n"
        "    error:\n"
        "      type: AutoPicassoError\n"
        "      message: kaboom\n"
        "      traceback: |\n"
        "        Traceback (most recent call last):\n"
        "          boom line\n"
    )
    window._monitor_local_folder = str(tmp_path)
    window._update_error_details([_failed_state(error=None)])
    text = window.error_details_display.toPlainText()
    assert "boom line" in text


def test_log_tail_fallback(window, tmp_path):
    """A failed run without any recorded module failure (e.g. an import
    error before the workflow started) shows the local log tail."""
    (tmp_path / "local_run.log").write_text("ImportError: nope\n")
    window._monitor_local_folder = str(tmp_path)
    window._local_process = None
    window._update_error_details(
        [
            _failed_state(
                modules=[{"i": 0, "name": "load", "status": "pending"}]
            )
        ]
    )
    text = window.error_details_display.toPlainText()
    assert "ImportError: nope" in text
    assert "local_run.log" in text


def test_yaml_string_error_fallback(window, tmp_path):
    """A yaml error recorded as a plain string (module-set, or old runs)
    must be shown, not crash the collection (err.get on a str used to
    raise and silently blank the whole pane)."""
    run_dir = tmp_path / "rep_123456-0000"
    run_dir.mkdir()
    (run_dir / "WorkflowRunner.yaml").write_text(
        "results:\n"
        "  01_identify:\n"
        "    success: false\n"
        "    error: something went sideways\n"
    )
    window._monitor_local_folder = str(tmp_path)
    window._update_error_details([_failed_state(error=None)])
    assert (
        "something went sideways" in window.error_details_display.toPlainText()
    )


def test_collection_crash_does_not_blank_pane(window, tmp_path, monkeypatch):
    """If collecting one module's details raises, the pane still shows the
    failure header plus a note instead of staying empty."""
    window._monitor_local_folder = str(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("yaml exploded")

    monkeypatch.setattr(window, "_local_yaml_error", boom)
    window._update_error_details([_failed_state(error=None)])
    text = window.error_details_display.toPlainText()
    assert "identify" in text
    assert "could not be collected" in text
    assert "yaml exploded" in text


def test_slurm_kill_reason_shown(window):
    """A job SLURM ended (e.g. OOM-killed) leaves no Python traceback, so
    the pane shows the SLURM reason and the module that was running."""
    window._monitor_local_folder = None
    slurm = {
        "success": True,
        "status": "OUT_OF_MEMORY",
        "details": {"exit_code": "0:125", "max_rss": "49.9G"},
    }
    state = {
        "kind": "single",
        "report_name": "rep_123456-0000",
        "state": "running",
        "current": 1,
        "total": 2,
        "modules": [
            {"i": 0, "name": "load", "status": "done"},
            {"i": 1, "name": "picasso_frc", "status": "running"},
        ],
    }
    window._update_error_details([state], slurm)
    text = window.error_details_display.toPlainText()
    assert "OUT_OF_MEMORY" in text
    assert "picasso_frc" in text
    assert "MaxRSS 49.9G" in text
    assert "Memory" in text  # the OOM hint


def test_slurm_completed_does_not_fill_pane(window):
    window._monitor_local_folder = None
    window.error_details_display.clear()
    slurm = {"success": True, "status": "COMPLETED", "details": {}}
    window._update_error_details([], slurm)
    assert window.error_details_display.toPlainText() == ""
