"""The bundled self-update: an all-users (read-only) install updates ELEVATED
(UAC) rather than failing silently, and the detached updater's outcome is
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


def _confirm_yes(monkeypatch):
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes),
    )


def test_self_update_elevates_on_readonly_install(window, monkeypatch):
    # read-only (all-users) install -> update must request elevation (UAC)
    monkeypatch.setattr(self_update, "install_dir_writable", lambda: False)
    _confirm_yes(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        self_update,
        "spawn_detached_updater",
        lambda url, parent_pid, elevate=False: captured.update(
            url=url, elevate=elevate
        ),
    )
    monkeypatch.setattr(window, "close", lambda: None)
    window._on_self_update_result(True, "https://x/pw.whl")
    assert captured["elevate"] is True
    assert captured["url"] == "https://x/pw.whl"


def test_self_update_no_elevation_when_writable(window, monkeypatch):
    # Just-Me install (user owns the files) -> no UAC, plain detached updater
    monkeypatch.setattr(self_update, "install_dir_writable", lambda: True)
    _confirm_yes(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        self_update,
        "spawn_detached_updater",
        lambda url, parent_pid, elevate=False: captured.update(
            elevate=elevate
        ),
    )
    monkeypatch.setattr(window, "close", lambda: None)
    window._on_self_update_result(True, "https://x/pw.whl")
    assert captured["elevate"] is False


def test_self_update_declined_elevation_keeps_app_open(window, monkeypatch):
    # UAC cancelled -> spawn raises -> app stays open, user is told
    monkeypatch.setattr(self_update, "install_dir_writable", lambda: False)
    _confirm_yes(monkeypatch)

    def _raise(*a, **k):
        raise OSError("declined")

    monkeypatch.setattr(self_update, "spawn_detached_updater", _raise)
    closed = {"v": False}
    monkeypatch.setattr(window, "close", lambda: closed.update(v=True))
    box = {}
    monkeypatch.setattr(
        window,
        "_update_message_box",
        lambda title, html, warning=False: box.update(
            title=title, warning=warning
        ),
    )
    window._on_self_update_result(True, "https://x/pw.whl")
    assert closed["v"] is False  # not closed into a no-op
    assert box["warning"] is True


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
