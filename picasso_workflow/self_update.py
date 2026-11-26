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


def update_log_path(prefix=None):
    """Where the detached updater writes its result.

    Defaults to a **per-user** dir, not the install prefix: an all-users install
    elevates the updater, which would then write the log into the read-only
    ``C:\\ProgramData`` prefix where the (unelevated) GUI could read but not
    delete it -- so the result would re-show every launch. A per-user path is
    writable and deletable by this user whether or not the updater ran elevated.
    ``prefix`` overrides (tests / explicit callers).
    """
    if prefix:
        return str(Path(prefix) / "update.log")
    try:
        from picasso_workflow import _user_config_dir

        base = Path(_user_config_dir())
        base.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - fall back to the env root
        base = Path(sys.prefix)
    return str(base / "update.log")


def install_dir_writable(target=None):
    """True if the dir pip installs into is writable by the current user now.

    A one-click **all-users** install lands under ``%ALLUSERSPROFILE%``
    (``C:\\ProgramData``), where a non-elevated user has read-only access -- so
    an in-place ``pip`` upgrade silently fails (Access Denied overwriting
    site-packages) and the version never changes. Probe by creating + removing a
    temp file in the install dir (``os.access`` is unreliable on Windows ACLs).
    ``target`` defaults to the env's ``purelib`` (where the wheel installs).
    """
    import sysconfig
    import tempfile

    target = target or sysconfig.get_paths().get("purelib") or sys.prefix
    try:
        fd, probe = tempfile.mkstemp(prefix=".pw-write-test-", dir=target)
    except OSError:
        return False
    os.close(fd)
    try:
        os.remove(probe)
    except OSError:  # pragma: no cover - best-effort cleanup
        pass
    return True


def read_update_outcome(prefix=None):
    """Read + consume the detached updater's result log, if present.

    Returns ``(status, detail)`` -- ``status`` is the first line
    (``OK`` / ``FAILED`` / ``SKIPPED``) and ``detail`` the rest (pip output or
    the skip reason) -- or ``None`` when no log exists. The log is removed after
    reading so the outcome is reported exactly once (on the next launch).
    """
    path = Path(update_log_path(prefix))
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - defensive against a locked file
        return None
    lines = text.splitlines()
    status = lines[0].strip() if lines else ""
    detail = "\n".join(lines[1:]).strip()
    try:
        path.unlink()
    except OSError:  # pragma: no cover - best-effort
        pass
    return status, detail


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


still_running = True
for _ in range(180):  # wait up to ~3 min for the GUI to close (DLLs unlock)
    if not _alive(parent_pid):
        still_running = False
        break
    time.sleep(1)
if still_running:
    # The app never closed -> its Qt/numpy DLLs are still locked, so pip would
    # fail (the exact thing we must avoid). Skip the install rather than leave a
    # half-updated env, and say so.
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("SKIPPED\n")
        fh.write("picasso-workflow was still running after the wait; the "
                 "update was not applied. Fully close it, then click "
                 "Update again.")
    sys.exit(0)
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
    wheel_url,
    parent_pid,
    python=None,
    log_path=None,
    *,
    popen=None,
    elevate=False,
    shell_execute=None,
):
    """Launch the detached updater to pip-install ``wheel_url`` after the GUI
    (``parent_pid``) exits.

    The helper is fully detached (own process group / session, no console) so it
    outlives the GUI closing; it waits for ``parent_pid`` to exit -- unlocking
    the loaded Qt/numpy DLLs -- then runs pip. Returns the log path it writes.

    ``elevate`` runs the helper with a Windows UAC prompt (``ShellExecute``
    ``runas``) -- required for an **all-users** install, whose program files
    under ``C:\\ProgramData`` are read-only to the user's default token (even an
    admin runs unelevated under UAC), so an unelevated pip can't overwrite them.
    Raises ``OSError`` if elevation is declined/fails. ``popen`` /
    ``shell_execute`` are injectable for tests.
    """
    python = python or sys.executable
    log_path = log_path or update_log_path()
    if elevate and os.name == "nt":
        _spawn_elevated_updater(
            python, wheel_url, parent_pid, log_path, shell_execute
        )
        return log_path
    popen = popen or subprocess.Popen
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


def _spawn_elevated_updater(
    python, wheel_url, parent_pid, log_path, shell_execute=None
):
    """Start the updater elevated via ``ShellExecuteW`` ``runas`` (UAC prompt).

    The script source is passed base64-encoded inside a ``python -c`` decoder so
    there is no temp file and no quoting of the multi-line source through the
    shell's parameter string. Raises ``OSError`` when the user declines UAC or
    elevation otherwise fails (``ShellExecuteW`` returns <= 32).
    """
    import base64

    blob = base64.b64encode(_UPDATER_SOURCE.encode("utf-8")).decode("ascii")
    # No spaces inside the quoted decoder except after `import`; the base64 blob
    # contains only [A-Za-z0-9+/=], so the single quotes around it are safe.
    decoder = (
        "import base64;" "exec(base64.b64decode('%s').decode('utf-8'))" % blob
    )
    params = '-c "%s" "%s" "%s" "%s"' % (
        decoder,
        wheel_url,
        parent_pid,
        log_path,
    )
    if shell_execute is None:
        import ctypes

        shell_execute = ctypes.windll.shell32.ShellExecuteW
    SW_HIDE = 0
    rc = shell_execute(None, "runas", python, params, None, SW_HIDE)
    if rc is not None and int(rc) <= 32:
        raise OSError(
            "Administrator elevation was declined or failed "
            f"(ShellExecute code {int(rc)})."
        )
