"""Headless engine for in-app self-update of a bundled (installer) install.

The one-click installer lays down a conda env with ``picasso_workflow`` + deps
pip-installed into a writable per-user prefix. For a *bundled* install we can
therefore update in place with ``pip`` -- no re-download of the whole installer
and no uninstall-first (constructor's NSIS installer cannot upgrade in place;
see the Windows upgrade notes). This module is Qt-free so it unit-tests without
a display; the GUI drives it and shows the restart instruction.

``picasso_workflow`` is not on PyPI, so the new wheel comes from the GitHub
release **assets** -- the same release :mod:`picasso_workflow.update_check`
already queries.

The update is applied while the app runs and takes effect on the **next
launch** (a restart instruction is surfaced by the caller); binary deps that are
already loaded can't be overwritten mid-run, so ``--upgrade-strategy
only-if-needed`` keeps dep churn to what the new wheel actually requires.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Marker file written into the install prefix by the installers' post_install,
# so self-update is offered only for a one-click-installer env (never a
# ``pip install -e`` dev checkout or a plain conda env).
BUNDLED_MARKER = ".picasso-workflow-bundled"


def is_bundled_install(prefix: str | None = None) -> bool:
    """True when running from a one-click-installer env (has the marker file).

    ``prefix`` defaults to ``sys.prefix`` (the running env root); overridable
    for tests.
    """
    base = Path(prefix) if prefix else Path(sys.prefix)
    return (base / BUNDLED_MARKER).exists()


def wheel_asset_url(release: dict) -> str | None:
    """Return the ``.whl`` asset download URL from a GitHub release, or None.

    ``release`` is the parsed GitHub "latest release" JSON (it carries an
    ``assets`` list with ``name`` + ``browser_download_url``).
    """
    for asset in (release or {}).get("assets", []) or []:
        if str(asset.get("name") or "").endswith(".whl"):
            url = asset.get("browser_download_url")
            if url:
                return str(url)
    return None


def pip_upgrade_command(wheel_url: str, python: str | None = None) -> list:
    """argv to upgrade ``picasso_workflow`` from ``wheel_url`` in this env.

    ``--upgrade-strategy only-if-needed`` upgrades a dependency only when the
    new wheel requires it, minimising overwrites of in-use binary deps while the
    app runs (the update applies on restart). Uses this env's interpreter
    (``sys.executable``) so pip targets the bundled env, not a system Python.
    """
    return [
        python or sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
        "--no-input",
        str(wheel_url),
    ]


def _looks_locked(text: str) -> bool:
    """Heuristic: did pip fail because a file is in use (app still running)?"""
    t = (text or "").lower()
    return (
        "being used by another process" in t
        or "access is denied" in t
        or "permission denied" in t
        or "winerror 32" in t
    )


def run_update(wheel_url, *, python=None, runner=None):
    """Run the pip upgrade and return ``(ok, message)``.

    ``message`` is user-facing: on success it tells the user to restart; on a
    locked-file failure it says to close the app / re-run the installer.
    ``runner`` is injectable for tests (defaults to :func:`subprocess.run`).
    """
    runner = runner or subprocess.run
    if not wheel_url:
        return False, "No update wheel was found in the latest release."
    proc = runner(
        pip_upgrade_command(wheel_url, python=python),
        capture_output=True,
        text=True,
    )
    if getattr(proc, "returncode", 1) == 0:
        return True, (
            "Update installed. Close and reopen picasso-workflow to finish "
            "applying it."
        )
    err = (
        getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or ""
    ).strip()
    if _looks_locked(err):
        return False, (
            "Some components are in use and could not be updated while the app "
            "is running. Close picasso-workflow, then re-run the installer for "
            "this update."
        )
    return False, "Update failed: " + (err or "unknown error")
