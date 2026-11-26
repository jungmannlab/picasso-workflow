"""The bundled self-update: a read-only (all-users) install is refused with a
clear message instead of a silent no-op, and the detached updater's outcome is
surfaced on the next launch."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

from PyQt6 import QtWidgets  # noqa: E402

from picasso_workflow import gui, self_update  # noqa: E402


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


def test_self_update_refused_when_install_dir_readonly(window, monkeypatch):
    # all-users / read-only install: pip can't overwrite site-packages
    monkeypatch.setattr(self_update, "install_dir_writable", lambda: False)
    captured = {}
    monkeypatch.setattr(
        window,
        "_update_message_box",
        lambda title, html, warning=False: captured.update(
            title=title, html=html, warning=warning
        ),
    )
    # must NOT spawn a worker/updater
    monkeypatch.setattr(
        self_update,
        "spawn_detached_updater",
        lambda *a, **k: pytest.fail("updater spawned on read-only install"),
    )
    window._update_thread = None
    window._start_self_update()
    assert "all users" in captured["html"].lower()
    assert captured["warning"] is True
    assert window._update_thread is None  # no background work started


def test_prior_update_outcome_surfaced_and_consumed(window, monkeypatch):
    monkeypatch.setattr(
        self_update,
        "read_update_outcome",
        lambda: ("FAILED", "ERROR: Access is denied: ...site-packages"),
    )
    captured = {}
    monkeypatch.setattr(
        window,
        "_update_message_box",
        lambda title, html, warning=False: captured.update(
            title=title, html=html, warning=warning
        ),
    )
    window._report_prior_update_outcome()
    assert captured["title"] == "Update failed"
    assert captured["warning"] is True
    assert "Access is denied" in captured["html"]
