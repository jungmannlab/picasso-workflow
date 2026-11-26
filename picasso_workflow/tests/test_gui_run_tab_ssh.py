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
