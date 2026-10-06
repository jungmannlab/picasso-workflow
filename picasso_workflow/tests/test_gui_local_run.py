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


# ---------------------------------------------------------------------------
# local-run monitoring: stale-state scoping and process-exit fusion
# ---------------------------------------------------------------------------


def _fake_proc(rc):
    class P:
        pid = 777

        def poll(self):
            return rc

    return P()


def test_scope_states_passthrough_without_launch(window):
    window._local_run_started_dt = None
    states = [{"updated": "2020-01-01T00:00:00"}]
    assert window._scope_states_to_local_launch(states) == states


def test_scope_states_drops_stale(window):
    from datetime import datetime, timedelta

    window._local_run_started_dt = datetime.now()
    old = {
        "updated": (datetime.now() - timedelta(hours=2)).isoformat(
            timespec="seconds"
        )
    }
    new = {"updated": datetime.now().isoformat(timespec="seconds")}
    unparsable = {"updated": None}
    kept = window._scope_states_to_local_launch([old, new, unparsable])
    assert new in kept
    assert old not in kept
    assert unparsable in kept  # unparsable: keep rather than hide


def test_dead_process_without_states_shows_failed_badge(window, tmp_path):
    """A local process that died before writing any progress must show as
    failed, not stay 'running' forever."""
    window._monitor_local_folder = str(tmp_path)
    window._local_process = _fake_proc(1)
    window._update_monitor_display(None, [])
    assert "failed (exit 1)" in window.monitor_state_label.text()


def test_dead_process_exit0_shows_finished(window, tmp_path):
    window._monitor_local_folder = str(tmp_path)
    window._local_process = _fake_proc(0)
    window._update_monitor_display(None, [])
    assert "finished" in window.monitor_state_label.text()


def test_dead_process_shows_log_tail_in_error_details(window, tmp_path):
    (tmp_path / "local_run.log").write_text(
        "ConfluenceInterfaceError: 403 FORBIDDEN\n"
    )
    window._monitor_local_folder = str(tmp_path)
    window._local_process = _fake_proc(1)
    window._update_monitor_display(None, [])
    assert "403 FORBIDDEN" in window.error_details_display.toPlainText()


def test_monitor_stops_when_process_died_without_states(window, tmp_path):
    window._monitor_local_folder = str(tmp_path)
    window._local_process = _fake_proc(1)
    window.monitor_timer.start()
    window._maybe_stop_monitor(None, [])
    assert not window.monitor_timer.isActive()
