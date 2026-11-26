"""The Run-tab 'Configure / Test connection…' button targets the SELECTED
login node, not the first configured one."""

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


def test_run_ssh_returns_result_and_clears_busy(window):
    # off-thread call returns the value; status bar + busy flag are cleared
    assert window._run_ssh("Connecting to login node", lambda: 7) == 7
    assert window._ssh_busy is False
    assert window.statusBar().currentMessage() == ""


def test_run_ssh_propagates_exception(window):
    def boom():
        raise RuntimeError("ssh blew up")

    with pytest.raises(RuntimeError, match="ssh blew up"):
        window._run_ssh("Connecting to login node", boom)
    assert window._ssh_busy is False  # cleaned up even on error


def test_run_ssh_reentry_runs_inline(window):
    # already busy (e.g. monitor timer firing mid-call) -> run inline, no hang
    window._ssh_busy = True
    try:
        assert window._run_ssh("x", lambda: 5) == 5
    finally:
        window._ssh_busy = False


def test_login_node_probe_surfaces_ssh_error(window):
    """The pre-flight probe returns the real ssh stderr on failure, uses
    batch_mode (fail-fast, no password hang), and is clean on success."""

    class _Comm:
        def __init__(self, result):
            self._result = result
            self.calls = []

        def execute_ssh_command(self, command, batch_mode=False):
            self.calls.append((command, batch_mode))
            return self._result

    ok_comm = _Comm({"success": True, "stderr": ""})
    assert window._login_node_probe(ok_comm) == (True, "")
    # batch_mode must be set so a bad key fails fast instead of hanging
    assert ok_comm.calls[0][1] is True

    fail_comm = _Comm(
        {"success": False, "stderr": "Permission denied (publickey).\n"}
    )
    ok, detail = window._login_node_probe(fail_comm)
    assert ok is False
    assert detail == "Permission denied (publickey)."


def test_run_tab_ssh_targets_selected_login_node(window, monkeypatch):
    monkeypatch.setitem(
        gui.CONFIG,
        "SlurmLoginNodes",
        {"clusterA": "login01", "clusterB": "login02"},
    )
    combo = window.cluster_host_combo
    combo.clear()
    combo.addItem("clusterA")
    combo.addItem("clusterB")
    combo.setCurrentText("clusterB")

    captured = {}
    monkeypatch.setattr(
        window,
        "on_configure_ssh",
        lambda default_host=None, default_label=None: captured.update(
            host=default_host, label=default_label
        ),
    )
    window._configure_ssh_from_run_tab()
    # the dialog is opened for clusterB's login-node hostname, not clusterA's
    assert captured["host"] == "login02"
    # ...and saves back to the cluster KEY (clusterB), so testing login02 does
    # NOT create a spurious "login02" host entry in the dropdown.
    assert captured["label"] == "clusterB"


def test_ssh_dialog_saves_under_cluster_key_not_hostname(qapp):
    # Given an existing cluster key as the label, the dialog saves under that
    # key (updates its login node) rather than under the hostname.
    dialog = gui.SshSetupDialog(
        default_host="login02", default_label="clusterB"
    )
    try:
        assert dialog.label_edit.text() == "clusterB"
        assert dialog.host_edit.text() == "login02"
        # the persisted label is the cluster key, not the hostname
        assert dialog._current_label() == "clusterB"
    finally:
        dialog.close()
