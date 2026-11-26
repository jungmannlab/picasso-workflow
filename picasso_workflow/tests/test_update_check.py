#!/usr/bin/env python
"""Tests for the GUI's GitHub "newer release?" check (pure logic, no network).

The network call is exercised by monkeypatching ``urllib.request.urlopen`` so
the suite stays offline-safe and deterministic.
"""

import io
import json

from picasso_workflow import update_check
from picasso_workflow.update_check import (
    UpdateInfo,
    _is_newer,
    check_for_update,
    fetch_latest_release,
)

# -- version comparison -------------------------------------------------------


def test_is_newer_basic():
    assert _is_newer("0.6.0", "0.5.6") is True
    assert _is_newer("0.5.6", "0.6.0") is False
    assert _is_newer("0.5.6", "0.5.6") is False


def test_is_newer_handles_dev_and_v_prefix():
    # a setuptools-scm dev build is a pre-release of its base version, so the
    # released base version counts as newer.
    assert _is_newer("0.5.6", "0.5.6.dev3+gabcdef") is True
    # a leading "v" is stripped by the caller; the comparator sees plain nums.
    assert _is_newer("0.6.0", "0.5.6") is True


def test_is_newer_bad_input_is_false():
    assert _is_newer("not-a-version", "0.5.6") is False


def test_is_newer_unknown_current_still_nudges():
    # When _version.py is missing, __version__ == "unknown"; a real release
    # must still count as newer so the user gets the update banner.
    assert _is_newer("0.6.0", "unknown") is True


# -- check_for_update (mocked HTTP) -------------------------------------------


def _fake_urlopen(payload):
    """Return a fake urlopen() yielding ``payload`` (dict) as a JSON body."""

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def _open(req, timeout=None):
        return _Resp(json.dumps(payload).encode("utf-8"))

    return _open


def test_check_returns_update_when_newer(monkeypatch):
    monkeypatch.setattr(
        update_check.urllib.request,
        "urlopen",
        _fake_urlopen(
            {
                "tag_name": "v0.9.0",
                "html_url": "https://example/releases/0.9.0",
            }
        ),
    )
    info = check_for_update("0.5.6")
    assert isinstance(info, UpdateInfo)
    assert info.latest == "0.9.0"
    assert info.url == "https://example/releases/0.9.0"


def test_check_returns_none_when_current(monkeypatch):
    monkeypatch.setattr(
        update_check.urllib.request,
        "urlopen",
        _fake_urlopen(
            {
                "tag_name": "v0.5.6",
                "html_url": "https://example/releases/0.5.6",
            }
        ),
    )
    assert check_for_update("0.5.6") is None


def test_check_returns_none_on_missing_fields(monkeypatch):
    monkeypatch.setattr(
        update_check.urllib.request,
        "urlopen",
        _fake_urlopen({"name": "no tag here"}),
    )
    assert check_for_update("0.5.6") is None


def test_check_never_raises_on_network_error(monkeypatch):
    def _boom(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(update_check.urllib.request, "urlopen", _boom)
    assert check_for_update("0.5.6") is None


# -- fetch_latest_release (3-way, for the on-demand dialog) -------------------


def test_fetch_latest_release_update(monkeypatch):
    monkeypatch.setattr(
        update_check.urllib.request,
        "urlopen",
        _fake_urlopen(
            {"tag_name": "v0.9.0", "html_url": "https://example/r/0.9.0"}
        ),
    )
    r = fetch_latest_release("0.5.6")
    assert r.status == "update"
    assert r.current == "0.5.6" and r.latest == "0.9.0"
    assert r.url == "https://example/r/0.9.0" and r.error is None


def test_fetch_latest_release_current(monkeypatch):
    monkeypatch.setattr(
        update_check.urllib.request,
        "urlopen",
        _fake_urlopen(
            {"tag_name": "v0.5.6", "html_url": "https://example/r/0.5.6"}
        ),
    )
    r = fetch_latest_release("0.5.6")
    assert r.status == "current" and r.latest == "0.5.6"


def test_fetch_latest_release_error_on_missing_fields(monkeypatch):
    monkeypatch.setattr(
        update_check.urllib.request,
        "urlopen",
        _fake_urlopen({"name": "no tag"}),
    )
    r = fetch_latest_release("0.5.6")
    assert r.status == "error" and r.latest is None


def test_fetch_latest_release_error_on_network(monkeypatch):
    def _boom(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(update_check.urllib.request, "urlopen", _boom)
    r = fetch_latest_release("0.5.6")
    assert r.status == "error"
    assert "network down" in (r.error or "")


def test_windows_upgrade_note_on_windows():
    note = update_check.windows_upgrade_note(platform="win32")
    assert note
    assert "uninstall" in note.lower()
    assert "same folder" in note.lower()


def test_windows_upgrade_note_empty_off_windows():
    assert update_check.windows_upgrade_note(platform="linux") == ""
    assert update_check.windows_upgrade_note(platform="darwin") == ""
