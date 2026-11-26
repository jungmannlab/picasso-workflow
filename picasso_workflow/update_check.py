"""Lightweight "is there a newer release?" check for the GUI.

Queries the GitHub Releases API for the latest published tag and compares it to
the running version. This module is pure logic (HTTP + version compare) so it
can be unit-tested without Qt; the GUI wiring (background thread + banner) lives
in :mod:`picasso_workflow.gui`.

Everything fails soft: offline, rate-limited, or malformed responses just yield
"no update", so the check never blocks startup or bothers the user.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from typing import NamedTuple

# The public repository whose Releases page carries the one-click installers.
GITHUB_REPO = "jungmannlab/picasso-workflow"
RELEASES_API_URL = (
    f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
)
# Human-facing releases page -- the fall-back link shown in the on-demand
# "Check for updates" dialog (e.g. when the check couldn't reach the API).
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_REPO}/releases"
# Short timeout: a slow / unreachable API must not delay the check (it runs off
# the GUI thread, but we still don't want it lingering).
_TIMEOUT_S = 4.0


def windows_upgrade_note(platform: str | None = None) -> str:
    """Windows-only caveat on how to upgrade over an existing install.

    constructor's NSIS installer aborts on a non-empty target directory and has
    no in-place-upgrade path, so a Windows user must uninstall the current
    version before installing the new one into the same folder. Returns an
    empty string off Windows. ``platform`` overrides ``sys.platform`` (tests);
    pass it explicitly (incl. ``""``) to force the result regardless of host.
    """
    effective = sys.platform if platform is None else platform
    if effective != "win32":
        return ""
    return (
        "On Windows, uninstall the current version first (Settings → "
        "Apps, or run Uninstall-picasso-workflow.exe in the install folder), "
        "then run the new installer into the same folder."
    )


class UpdateInfo(NamedTuple):
    """A newer release than the one running."""

    latest: str  # version string of the latest release, e.g. "0.6.0"
    url: str  # the release's human-facing html_url


class CheckResult(NamedTuple):
    """Outcome of an on-demand release check (for the menu dialog).

    Unlike :class:`UpdateInfo` (returned only when newer), this reports all
    three outcomes so the dialog can distinguish "up to date" from "couldn't
    check".
    """

    status: str  # "update" | "current" | "error"
    current: str  # the running version (v-stripped)
    latest: str | None  # latest tag, or None on error
    url: str | None  # the release html_url, or None on error
    error: str | None  # error text when status == "error"


def _is_newer(latest: str, current: str) -> bool:
    """Return True if ``latest`` is a strictly newer version than ``current``.

    Prefers PEP 440 comparison via ``packaging`` (handles the
    setuptools-scm dev/local versions this project uses); falls back to a naive
    numeric-tuple compare when ``packaging`` is unavailable or a version is
    unparseable (e.g. the ``"unknown"`` fallback when ``_version.py`` is
    missing -- in which case we still want any real release to count as newer
    so the user gets nudged).
    """

    def _nums(v):
        return tuple(int(x) for x in re.findall(r"\d+", v)[:3])

    try:
        from packaging.version import InvalidVersion, parse

        try:
            return parse(latest) > parse(current)
        except InvalidVersion:
            return _nums(latest) > _nums(current)
    except Exception:
        return _nums(latest) > _nums(current)


def fetch_latest_release(
    current_version: str,
    url: str = RELEASES_API_URL,
    timeout: float = _TIMEOUT_S,
) -> CheckResult:
    """Query GitHub's latest release; return a 3-way :class:`CheckResult`.

    Distinguishes "update" / "current" / "error" so the on-demand dialog can
    report which happened (:func:`check_for_update` collapses the last two to
    ``None`` for the passive startup banner). Never raises.

    Parameters
    ----------
    current_version : str
        The running version (e.g. ``picasso_workflow.__version__``). A leading
        ``v`` is tolerated.
    url : str
        GitHub "latest release" API endpoint; overridable for tests.
    timeout : float
        Per-request timeout in seconds.
    """
    current = str(current_version).lstrip("v")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "picasso-workflow",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        tag = str(data.get("tag_name") or "").lstrip("v")
        html_url = str(data.get("html_url") or "")
        if not tag or not html_url:
            return CheckResult(
                "error", current, None, None, "malformed GitHub response"
            )
        if _is_newer(tag, current):
            return CheckResult("update", current, tag, html_url, None)
        return CheckResult("current", current, tag, html_url, None)
    except Exception as exc:  # fails soft -- report, don't raise
        return CheckResult("error", current, None, None, str(exc))


def fetch_latest_release_json(
    url: str = RELEASES_API_URL, timeout: float = _TIMEOUT_S
) -> dict | None:
    """Return the parsed GitHub "latest release" JSON, or ``None`` on any error.

    Unlike :func:`fetch_latest_release` (which distils a few fields), this hands
    back the raw release object -- the ``assets`` list lets the in-app updater
    find the downloadable wheel. Fails soft (never raises).
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "picasso-workflow",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # fails soft -- caller treats None as "no update"
        return None


def check_for_update(
    current_version: str,
    url: str = RELEASES_API_URL,
    timeout: float = _TIMEOUT_S,
) -> UpdateInfo | None:
    """Return :class:`UpdateInfo` if a newer release exists, else ``None``.

    Never raises -- any error (network, HTTP, JSON, version parse) is treated
    as "no update available". Thin wrapper over :func:`fetch_latest_release`
    for the passive startup banner.

    Parameters
    ----------
    current_version : str
        The running version (e.g. ``picasso_workflow.__version__``). A leading
        ``v`` is tolerated.
    url : str
        GitHub "latest release" API endpoint; overridable for tests.
    timeout : float
        Per-request timeout in seconds.
    """
    result = fetch_latest_release(current_version, url=url, timeout=timeout)
    if result.status == "update":
        return UpdateInfo(latest=result.latest, url=result.url)
    return None
