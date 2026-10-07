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


def test_monitor_stops_after_kill_with_running_state(window, tmp_path):
    """After 'Kill local run' the killed process leaves progress.json
    non-terminal ('running'); the monitor must still stop polling because
    the process we launched has exited."""
    window._monitor_local_folder = str(tmp_path)
    window._local_process = _fake_proc(-15)  # terminated by signal
    window.monitor_timer.start()
    running = [{"kind": "single", "state": "running", "modules": []}]
    window._maybe_stop_monitor(None, running)
    assert not window.monitor_timer.isActive()


def test_stale_job_id_does_not_flip_live_local_run(window, tmp_path):
    """A leftover cluster Job-ID must not flip a live local run's monitor
    back to cluster mode."""
    (tmp_path / "progress.json").write_text("{}")
    window.results_folder_display.setText(str(tmp_path))
    window.job_id_input.setText("123456")  # stale cluster job id
    window._local_process = _fake_proc(None)  # local run still alive
    window._monitor_local_folder = str(tmp_path)
    window._resolve_monitor_target()
    assert window._monitor_local_folder == str(tmp_path)  # stayed local


def test_repoint_resets_launch_state(window, tmp_path):
    """Pointing the monitor at a different folder (no live run) drops the
    previous launch's scoping/process so the new folder's runs show."""
    from datetime import datetime

    other = tmp_path / "other"
    other.mkdir()
    (other / "progress.json").write_text("{}")
    window._monitor_local_folder = str(tmp_path / "old")
    window._local_run_started_dt = datetime.now()
    window._local_process = _fake_proc(1)  # dead previous run
    window.job_id_input.clear()
    window.results_folder_display.setText(str(other))
    window._resolve_monitor_target()
    assert window._monitor_local_folder == str(other)
    assert window._local_run_started_dt is None
    assert window._local_process is None
