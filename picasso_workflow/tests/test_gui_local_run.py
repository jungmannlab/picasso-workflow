#!/usr/bin/env python
"""GUI tests for the "Run locally" tab.

The tab must be enabled and carry working controls: graceful stop (abort
flag), kill, and a log-tail viewer. The live monitor and run-information
display are shared with cluster runs (they live below the run sub-tabs).

Skips where a Qt GUI cannot be constructed.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

from PyQt6 import QtWidgets  # noqa: E402

from picasso_workflow import gui  # noqa: E402
from picasso_workflow import progress as pwprogress  # noqa: E402


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


def test_local_tab_enabled(window):
    assert window.run_tabs.tabText(1) == "Run locally"
    assert window.run_tabs.isTabEnabled(1)


def test_shared_monitor_and_info_exist(window):
    # the monitor and run-information display are built once, shared
    # between the cluster and local run modes
    assert window.monitor_state_label is not None
    assert window.module_tree is not None
    assert window.job_info_display is not None


def test_stop_local_without_run_reports(window):
    window._monitor_local_folder = None
    window.on_stop_local_run()
    assert "No local run to stop" in window.job_info_display.toPlainText()


def test_stop_local_run_writes_abort_flag(window, tmp_path):
    window._monitor_local_folder = str(tmp_path)
    window.on_stop_local_run()
    assert pwprogress.abort_requested(str(tmp_path))
    assert "Abort requested" in window.job_info_display.toPlainText()


def test_kill_local_without_process_reports(window):
    window._local_process = None
    window.on_kill_local_run()
    assert "No running local process" in window.job_info_display.toPlainText()


def test_show_local_log_tail(window, tmp_path):
    (tmp_path / "local_run.log").write_text("line one\nline two\n")
    window._monitor_local_folder = str(tmp_path)
    window.on_show_local_log()
    text = window.job_info_display.toPlainText()
    assert "line two" in text


def test_show_local_log_missing_reports(window, tmp_path):
    window._monitor_local_folder = str(tmp_path / "nowhere")
    window.on_show_local_log()
    assert "Could not read" in window.job_info_display.toPlainText()


def test_start_locally_refuses_second_run(window):
    class FakeProc:
        pid = 4242

        def poll(self):
            return None  # still running

    window._local_process = FakeProc()
    window.start_locally()
    assert "already active (PID 4242)" in window.job_info_display.toPlainText()
