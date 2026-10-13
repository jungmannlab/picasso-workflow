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
import urllib.request
from typing import NamedTuple

# The public repository whose Releases page carries the one-click installers.
GITHUB_REPO = "jungmannlab/picasso-workflow"
RELEASES_API_URL = (
    f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
)
# Short timeout: a slow / unreachable API must not delay the check (it runs off
# the GUI thread, but we still don't want it lingering).
_TIMEOUT_S = 4.0


class UpdateInfo(NamedTuple):
    """A newer release than the one running."""

    latest: str  # version string of the latest release, e.g. "0.6.0"
    url: str  # the release's human-facing html_url


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


def check_for_update(
    current_version: str,
    url: str = RELEASES_API_URL,
    timeout: float = _TIMEOUT_S,
) -> UpdateInfo | None:
    """Return :class:`UpdateInfo` if a newer release exists, else ``None``.

    Never raises -- any error (network, HTTP, JSON, version parse) is treated
    as "no update available".

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
            return None
        current = str(current_version).lstrip("v")
        if _is_newer(tag, current):
            return UpdateInfo(latest=tag, url=html_url)
    except Exception:
        return None
    return None
