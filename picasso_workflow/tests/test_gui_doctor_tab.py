#!/usr/bin/env python
"""Tests for the GUI "Doctor" tab (diagnostics + connection probe)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

from PyQt6 import QtWidgets  # noqa: E402

from picasso_workflow.doctor_tab import (  # noqa: E402
    DoctorTab,
    collect_diagnostics,
    probe_confluence,
    _env_status,
)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# -- pure helpers -------------------------------------------------------------


def test_env_status_masks_secrets(monkeypatch):
    # a secret-named var is never shown as its value
    monkeypatch.setenv("CONFLUENCE_TOKEN", "super-secret-value")
    assert _env_status("CONFLUENCE_TOKEN") == "set"
    monkeypatch.delenv("CONFLUENCE_TOKEN", raising=False)
    assert _env_status("CONFLUENCE_TOKEN") == "not set"
    # a non-secret var shows its value
    monkeypatch.setenv("CONFLUENCE_URL", "https://conf.example/wiki")
    assert _env_status("CONFLUENCE_URL") == "https://conf.example/wiki"


def test_collect_diagnostics_shape():
    diag = collect_diagnostics()
    assert set(diag) >= {
        "versions",
        "cluster",
        "config_sources",
        "dotenv",
        "logging",
        "env",
        "confluence",
    }
    assert diag["versions"]["picasso_workflow"]
    # both Confluence profiles summarised, with presence-only token flag
    assert set(diag["confluence"]) == {"Confluence", "ConfluenceTest"}
    for s in diag["confluence"].values():
        assert "token_set" in s and isinstance(s["token_set"], bool)
        assert "token" not in s  # never the value


def _patch_creds(monkeypatch, **over):
    creds = {
        "base_url": "https://conf.example/wiki",
        "space_key": "SPACE",
        "parent_page_title": "Home",
        "username": None,
        "token": "tok",
    }
    creds.update(over)
    monkeypatch.setattr(
        "picasso_workflow.confluence.resolve_confluence_credentials",
        lambda profile: creds,
    )


class _FakeClient:
    def __init__(self, page):
        self._page = page

    def get_page_by_title(self, space, title):
        if isinstance(self._page, Exception):
            raise self._page
        return self._page


def _patch_iface(monkeypatch, page):
    class _FakeIface:
        def __init__(self, **kwargs):
            # the constructor must not do a title lookup -> dummy page id
            assert kwargs.get("parent_page_id")
            self.confluence = _FakeClient(page)

    monkeypatch.setattr(
        "picasso_workflow.confluence.ConfluenceInterface", _FakeIface
    )


def test_probe_confluence_no_token(monkeypatch):
    _patch_creds(monkeypatch, token=None)
    status, msg = probe_confluence("Confluence")
    assert status == "fail" and "token" in msg.lower()


def test_probe_confluence_ok(monkeypatch):
    _patch_creds(monkeypatch)
    _patch_iface(monkeypatch, {"id": "42", "title": "Home"})
    status, msg = probe_confluence("Confluence")
    assert status == "ok" and "42" in msg


def test_probe_confluence_page_missing_is_warn(monkeypatch):
    _patch_creds(monkeypatch)
    _patch_iface(monkeypatch, None)
    status, msg = probe_confluence("Confluence")
    assert status == "warn" and "not found" in msg


def test_probe_confluence_request_failure_is_fail(monkeypatch):
    _patch_creds(monkeypatch)
    _patch_iface(monkeypatch, RuntimeError("401 Unauthorized"))
    status, msg = probe_confluence("Confluence")
    assert status == "fail" and "401" in msg


# -- the widget ---------------------------------------------------------------


def test_main_window_has_doctor_tab(qapp):
    from picasso_workflow import gui

    try:
        win = gui.Window()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not construct GUI window: {e}")
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Doctor" in titles
    win.close()


def test_doctor_tab_builds_and_renders(qapp):
    tab = DoctorTab()
    text = tab.info_view.toPlainText()
    assert "Versions" in text and "Confluence profiles" in text
    # a connection row per profile
    assert set(tab._conn_buttons) == {"Confluence", "ConfluenceTest"}
    assert tab._conn_results["Confluence"].text() == "not tested"
    tab.deleteLater()


def test_doctor_tab_connection_test_threaded(qapp, monkeypatch):
    """The test button runs the probe off-thread, updates the result, and the
    worker thread is cleaned up (no running QThread left to abort)."""
    monkeypatch.setattr(
        "picasso_workflow.doctor_tab.probe_confluence",
        lambda profile: ("ok", "all good"),
    )
    tab = DoctorTab()
    tab._test_connection("Confluence")
    # button disabled while in flight
    assert not tab._conn_buttons["Confluence"].isEnabled()

    import time

    for _ in range(50):
        qapp.processEvents()
        if tab._conn_results["Confluence"].text().startswith("OK"):
            break
        time.sleep(0.05)
    assert tab._conn_results["Confluence"].text().startswith("OK: all good")
    assert tab._conn_buttons["Confluence"].isEnabled()

    # the worker thread finished and was forgotten; stop any leftovers so no
    # live QThread outlives the test
    qapp.processEvents()
    tab._stop_threads()
    assert tab._threads == []
    tab.deleteLater()
