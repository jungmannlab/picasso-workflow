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
        lambda default_host=None: captured.update(host=default_host),
    )
    window._configure_ssh_from_run_tab()
    # the dialog is opened for clusterB's login-node hostname, not clusterA's
    assert captured["host"] == "login02"
