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

pip cannot overwrite the Qt/numpy DLLs the running GUI has loaded (and the
launcher imports them before any of our code runs, so "apply on next startup"
is out too). So the update runs in a **detached helper process** that waits for
the GUI to exit, then pip-installs -- see :func:`spawn_detached_updater`. The
app is not auto-relaunched; the user reopens it once the update finishes.
"""

from __future__ import annotations

import os
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


def wheel_asset_url(release: dict, version: str | None = None) -> str | None:
    """Return the ``picasso_workflow`` ``.whl`` asset URL from a release.

    ``release`` is the parsed GitHub "latest release" JSON (an ``assets`` list
    with ``name`` + ``browser_download_url``). Only a wheel whose (normalised)
    name is for ``picasso_workflow`` is considered, so a stray non-package wheel
    can't be picked. When ``version`` is given, **only** a wheel carrying that
    version is returned (else ``None``) -- so a botched release whose wheel
    doesn't match the advertised tag is refused rather than silently installed.
    """
    candidates = []
    for asset in (release or {}).get("assets", []) or []:
        name = str(asset.get("name") or "")
        if not name.endswith(".whl"):
            continue
        if not name.replace("-", "_").lower().startswith("picasso_workflow"):
            continue
        url = asset.get("browser_download_url")
        if url:
            candidates.append((name, str(url)))
    if not candidates:
        return None
    if version:
        # Wheel name is e.g. picasso_workflow-0.8.7-py3-none-any.whl; the
        # version is the 2nd '-'-delimited field -> exact token match.
        for name, url in candidates:
            if version in name.split("-"):
                return url
        return None
    return candidates[0][1]


def pip_upgrade_command(wheel_url: str, python: str | None = None) -> list:
    """argv to upgrade ``picasso_workflow`` from ``wheel_url`` in this env.

    ``--upgrade-strategy only-if-needed`` upgrades a dependency only when the
    new wheel requires it (keeps the update small). Uses the given interpreter
    (the bundled env's python), not a system Python. Shared by the detached
    updater's pip invocation.
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


def update_log_path(prefix=None):
    """Where the detached updater writes its result (`<prefix>/update.log`)."""
    base = Path(prefix) if prefix else Path(sys.prefix)
    return str(base / "update.log")


# Self-contained helper run as a DETACHED process: it waits for the GUI (whose
# loaded Qt/numpy DLLs would otherwise block pip from overwriting them) to exit,
# then pip-installs the wheel. It imports NO picasso_workflow code, so its own
# process never holds those DLLs. Kept as source so it runs under the env's
# python via `-c` -- no extra file to ship.
_UPDATER_SOURCE = r"""
import os, sys, time, subprocess
wheel, parent_pid, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]


def _alive(pid):
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


for _ in range(180):  # wait up to ~3 min for the GUI to close (DLLs unlock)
    if not _alive(parent_pid):
        break
    time.sleep(1)
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--upgrade",
     "--upgrade-strategy", "only-if-needed", "--no-input", wheel],
    capture_output=True, text=True,
)
with open(log, "w", encoding="utf-8") as fh:
    fh.write("OK\n" if r.returncode == 0 else "FAILED\n")
    fh.write((r.stdout or "") + "\n" + (r.stderr or ""))
"""


def spawn_detached_updater(
    wheel_url, parent_pid, python=None, log_path=None, *, popen=None
):
    """Launch the detached updater to pip-install ``wheel_url`` after the GUI
    (``parent_pid``) exits.

    The helper is fully detached (own process group / session, no console) so it
    outlives the GUI closing; it waits for ``parent_pid`` to exit -- unlocking
    the loaded Qt/numpy DLLs -- then runs pip. Returns the log path it writes.
    ``popen`` is injectable for tests (defaults to :class:`subprocess.Popen`).
    """
    popen = popen or subprocess.Popen
    python = python or sys.executable
    log_path = log_path or update_log_path()
    cmd = [
        python,
        "-c",
        _UPDATER_SOURCE,
        str(wheel_url),
        str(parent_pid),
        str(log_path),
    ]
    kwargs = {"close_fds": True}
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000
    else:
        kwargs["start_new_session"] = True
    popen(cmd, **kwargs)
    return log_path
