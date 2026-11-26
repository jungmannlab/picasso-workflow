#!/usr/bin/env python
"""The GUI "Doctor" tab: read-only diagnostics for the running GUI.

Surfaces, at a glance, the state that otherwise only lives in the logfile:
which ``config.yaml`` layers were loaded and from where, which ``.env`` was
picked up, where the log is written, the picasso-workflow / picasso / Python
versions, whether we are on a SLURM node, and -- per Confluence profile --
the resolved (non-secret) connection settings plus whether a token is set.

Everything shown is passive and gathered without side effects; **secrets are
never displayed as values**, only as "set" / "not set" (see ``_SECRET_ENV``).
The active features are explicit, opt-in, and run on a background thread (so
the GUI never blocks); no network/SSH call is made when the tab is merely
opened or refreshed:

* a *Test connection* button per Confluence profile (one authenticated
  request, reports OK / warning / failure); and
* a *Cluster secrets* panel that checks and provisions the Confluence token
  files (``~/.config/picasso_workflow/.env`` and ``~/.picasso_secrets``) on
  the Run-tab's cluster over SSH -- so the whole set-up can be done from the
  GUI. Token values travel only inside the written files (chmod 600), never
  on a command line, and are never displayed or logged.

Author: Heinrich Grabmayr
"""

from __future__ import annotations

import html
import os
import platform
import shlex
import sys

from loguru import logger
from PyQt6 import QtCore, QtWidgets

# Env vars that may hold a secret -> shown as presence only, never the value.
_SECRET_ENV = {
    "CONFLUENCE_TOKEN",
    "CONFLUENCE_BEARER",
    "TEST_CONFLUENCE_TOKEN",
}

# Env vars the tab surfaces. Non-secret ones show their value; the secret
# ones (``_SECRET_ENV``) show only "set" / "not set".
_ENV_VARS = [
    "CONFLUENCE_URL",
    "CONFLUENCE_SPACE",
    "CONFLUENCE_BASE_PAGE",
    "CONFLUENCE_USERNAME",
    "CONFLUENCE_TOKEN",
    "CONFLUENCE_BEARER",
    "TEST_CONFLUENCE_URL",
    "TEST_CONFLUENCE_SPACE",
    "TEST_CONFLUENCE_PAGE",
    "TEST_CONFLUENCE_USERNAME",
    "TEST_CONFLUENCE_TOKEN",
    "SLURM_JOB_ID",
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "PW_LOG_DIR",
    "DRIVEPATHS",
    "PYTHONNOUSERSITE",
]

# Confluence profiles to summarise / offer a connection test for.
_PROFILES = ("Confluence", "ConfluenceTest")

# Per-request timeout (seconds) for the connection test. Kept short so a dead
# or slow host can't hang the worker -- and, since app shutdown joins that
# worker, the GUI -- for atlassian's long default request timeout (75 s).
_PROBE_TIMEOUT_S = 10


def _env_status(name: str) -> str:
    """Describe an env var for display: its value, or "set"/"not set".

    Secret-named vars (``_SECRET_ENV``) never reveal their value.
    """
    value = os.environ.get(name)
    if name in _SECRET_ENV:
        return "set" if value else "not set"
    return "not set" if value is None else value


def _versions() -> dict:
    """picasso-workflow / picasso / Python / platform versions."""
    try:
        from picasso_workflow import __version__ as pw_version
    except Exception:  # pragma: no cover - defensive
        pw_version = "unknown"
    # read picasso's version only if it is already imported, to avoid pulling
    # its heavy import chain just for a diagnostics panel
    picasso_mod = sys.modules.get("picasso")
    if picasso_mod is None:
        picasso_version = "not loaded"
    else:
        picasso_version = getattr(picasso_mod, "__version__", "unknown")
    return {
        "picasso_workflow": pw_version,
        "picasso": picasso_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _confluence_summary(profile: str) -> dict:
    """Resolved non-secret Confluence settings + token presence for a profile.

    The token value is never returned -- only ``token_set``.
    """
    try:
        from picasso_workflow.confluence import (
            resolve_confluence_credentials,
        )

        creds = resolve_confluence_credentials(profile)
    except Exception as e:  # pragma: no cover - defensive
        return {"error": str(e)}
    return {
        "base_url": creds.get("base_url"),
        "space_key": creds.get("space_key"),
        "parent_page_title": creds.get("parent_page_title"),
        "username": creds.get("username"),
        "token_set": bool(creds.get("token")),
    }


def _test_profiles() -> list:
    """Profiles to offer a *connection test* for.

    The operational ``Confluence`` is always offered; a non-operational profile
    (``ConfluenceTest``, the pytest-suite target) only when it is actually
    configured -- it has a resolved ``base_url`` or a token. A typical user who
    never set up a test Confluence therefore doesn't see an irrelevant test
    button (the config/env templates no longer ship a ``ConfluenceTest``
    section, so it resolves to nothing for them).
    """
    profiles = ["Confluence"]
    for profile in _PROFILES:
        if profile == "Confluence":
            continue
        summary = _confluence_summary(profile)
        if summary.get("base_url") or summary.get("token_set"):
            profiles.append(profile)
    return profiles


def collect_diagnostics() -> dict:
    """Gather the read-only diagnostics shown by the tab (no secrets)."""
    import picasso_workflow as pw

    diag = getattr(pw, "DIAGNOSTICS", {}) or {}
    return {
        "versions": _versions(),
        "cluster": {"on_cluster": getattr(pw, "ON_CLUSTER", None)},
        "config_sources": diag.get("config_sources", []),
        "dotenv": diag.get("dotenv", {}),
        "user_env": diag.get("user_env", {}),
        "logging": diag.get("logging", {}),
        "env": {name: _env_status(name) for name in _ENV_VARS},
        # Only summarise a profile that's actually in play: the operational
        # Confluence always, a test profile (ConfluenceTest) only when it's
        # configured -- so a typical user's diagnostics aren't cluttered with an
        # unconfigured test Confluence (matches the Connection-tests section).
        "confluence": {p: _confluence_summary(p) for p in _test_profiles()},
    }


def probe_confluence(profile: str) -> tuple[str, str]:
    """Make one authenticated Confluence request and report the outcome.

    Builds a ``ConfluenceInterface`` with a dummy ``parent_page_id`` so its
    constructor does *not* perform the title lookup (a single, retry-free
    request is made here instead), then fetches the configured ``DefaultPage``
    to verify both authentication and that the page exists.

    Parameters
    ----------
    profile : str
        ``"Confluence"`` or ``"ConfluenceTest"``.

    Returns
    -------
    (status, message) : tuple of str
        ``status`` is ``"ok"`` / ``"warn"`` / ``"fail"``.
    """
    from picasso_workflow.confluence import (
        ConfluenceInterface,
        resolve_confluence_credentials,
    )

    creds = resolve_confluence_credentials(profile)
    if not creds.get("base_url"):
        return ("fail", "No base URL configured for this profile.")
    if not creds.get("token"):
        return ("fail", "No token set (expected in an environment variable).")

    try:
        iface = ConfluenceInterface(
            base_url=creds["base_url"],
            space_key=creds.get("space_key"),
            parent_page_title=creds.get("parent_page_title") or "",
            username=creds.get("username"),
            token=creds["token"],
            # skip the constructor's title lookup; probe explicitly below
            parent_page_id="__doctor_probe__",
        )
    except Exception as e:
        return ("fail", f"Could not connect: {e}")

    # bound each request so an unreachable host can't hang the worker (and,
    # since shutdown joins it, the GUI) for atlassian's 75 s default timeout
    try:
        iface.confluence.timeout = _PROBE_TIMEOUT_S
    except Exception:  # pragma: no cover - defensive (atlassian internals)
        pass

    title = creds.get("parent_page_title")
    space = creds.get("space_key")
    if not title:
        return (
            "warn",
            "Authenticated client built, but no DefaultPage configured to "
            "verify against.",
        )
    try:
        page = iface.confluence.get_page_by_title(space=space, title=title)
    except Exception as e:
        return ("fail", f"Authenticated request failed: {e}")
    if page is not None:
        return (
            "ok",
            f"Authenticated; DefaultPage '{title}' found (id {page['id']}).",
        )
    return (
        "warn",
        f"Authenticated, but DefaultPage '{title}' was not found in space "
        f"'{space}'.",
    )


class _ConnTestWorker(QtCore.QObject):
    """Runs :func:`probe_confluence` off the GUI thread."""

    finished = QtCore.pyqtSignal(str, str, str)  # profile, status, message

    def __init__(self, profile: str) -> None:
        super().__init__()
        self._profile = profile

    def run(self) -> None:
        try:
            status, message = probe_confluence(self._profile)
        except Exception as e:  # pragma: no cover - defensive
            status, message = "fail", f"Unexpected error: {e}"
        self.finished.emit(self._profile, status, message)


# --------------------------------------------------------------------------- #
# Cluster secrets provisioning (Doctor tab)
#
# The two files the cluster reads for the Confluence token, both under the
# remote $HOME:
#   ~/.config/picasso_workflow/.env  -- dotenv, auto-loaded by the package at
#                                       import (home is bound into the image).
#   ~/.picasso_secrets               -- `export KEY=VALUE`, `source`d by the
#                                       SLURM job scripts (and the test chain).
# Both are written chmod 600; token values travel only inside the scp'd file
# bodies, never on an ssh/scp command line, so they stay out of logs.
# --------------------------------------------------------------------------- #

# Remote shell expressions (the remote shell expands $HOME). The check command
# uses these directly; the upload resolves an absolute $HOME first (scp in
# SFTP mode does not expand ~).
_CLUSTER_ENV_REMOTE = '"$HOME/.config/picasso_workflow/.env"'
_CLUSTER_SECRETS_REMOTE = '"$HOME/.picasso_secrets"'


def _sh_squote(value: str) -> str:
    """Single-quote ``value`` for POSIX sh (safe inside a file we write)."""
    return "'" + value.replace("'", "'\\''") + "'"


def _secrets_file_content(op_token, test_token) -> str:
    """Build the ``~/.picasso_secrets`` text (``export`` lines, ``source``d).

    Only a provided (non-empty) token gets a line. Values are single-quoted so
    arbitrary token characters survive ``source``.
    """
    lines = [
        "# picasso-workflow secrets -- sourced by the SLURM job scripts.",
        "# Managed by the GUI Doctor tab; keep chmod 600.",
    ]
    if op_token:
        lines.append(f"export CONFLUENCE_TOKEN={_sh_squote(op_token)}")
    if test_token:
        lines.append(f"export TEST_CONFLUENCE_TOKEN={_sh_squote(test_token)}")
    return "\n".join(lines) + "\n"


def _env_file_content(op_token) -> str:
    """Build the ``~/.config/picasso_workflow/.env`` text (dotenv, auto-loaded).

    Confluence API tokens are URL-safe base64 (no spaces/quotes/#), so the
    value is written unquoted -- which python-dotenv reads back verbatim.
    """
    lines = [
        "# picasso-workflow per-user config -- auto-loaded at import.",
        "# Managed by the GUI Doctor tab; keep chmod 600.",
    ]
    if op_token:
        lines.append(f"CONFLUENCE_TOKEN={op_token}")
    return "\n".join(lines) + "\n"


def _file_check_snippet(label: str, remote_path: str) -> str:
    """One file's portion of the remote check (prints NO secret value).

    ``remote_path`` is a shell expression (e.g. ``"$HOME/.picasso_secrets"``);
    ``label`` is a fixed literal echoed back for parsing. Token presence is
    probed with ``grep -q`` only, so values never leave the cluster.
    """
    return (
        f"f={remote_path}; "
        'if [ -e "$f" ]; then '
        'r=no; [ -r "$f" ] && r=yes; '
        "p=$(stat -c '%a' \"$f\" 2>/dev/null || echo '?'); "
        "op=no; grep -qE '(^|[^A-Za-z0-9_])CONFLUENCE_(TOKEN|BEARER)=' "
        '"$f" 2>/dev/null && op=yes; '
        "te=no; grep -q 'TEST_CONFLUENCE_TOKEN=' \"$f\" 2>/dev/null "
        "&& te=yes; "
        f'echo "FILE|{label}|exists|readable=$r|perm=$p|op=$op|test=$te"; '
        f'else echo "FILE|{label}|missing"; fi'
    )


def _cluster_check_command() -> str:
    """Single remote command reporting both files' status (no secret value)."""
    return "; ".join(
        [
            _file_check_snippet("env", _CLUSTER_ENV_REMOTE),
            _file_check_snippet("secrets", _CLUSTER_SECRETS_REMOTE),
        ]
    )


def _parse_cluster_check(stdout: str) -> dict:
    """Parse the remote check output into ``{label: {...}}``.

    Labels are ``env`` and ``secrets``. Each maps to ``{"exists": False}`` when
    missing, else ``{"exists": True}`` plus ``readable``/``perm``/``op``/
    ``test`` string flags.
    """
    out: dict = {}
    for line in stdout.splitlines():
        if not line.startswith("FILE|"):
            continue
        parts = line.split("|")
        label = parts[1] if len(parts) > 1 else ""
        if len(parts) >= 3 and parts[2] == "missing":
            out[label] = {"exists": False}
            continue
        info: dict = {"exists": True}
        for kv in parts[3:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                info[k] = v
        out[label] = info
    return out


def _local_tokens():
    """Return ``(operational, test)`` Confluence tokens from local config.

    Each may be ``None``. Lazy import dodges the confluence import cycle.
    """
    try:
        from picasso_workflow.confluence import (
            resolve_confluence_credentials,
        )

        op = resolve_confluence_credentials("Confluence").get("token")
        test = resolve_confluence_credentials("ConfluenceTest").get("token")
        return op, test
    except Exception:  # pragma: no cover - defensive
        return None, None


def _format_cluster_check(parsed: dict) -> str:
    """Render a parsed cluster check as a small HTML status (for a QLabel)."""
    ok, warn, bad = "#2e7d32", "#b26a00", "#c62828"

    def mark(present: bool) -> str:
        return "✓" if present else "✗"

    rows = []
    for key, display in (
        ("env", "~/.config/picasso_workflow/.env"),
        ("secrets", "~/.picasso_secrets"),
    ):
        info = parsed.get(key)
        if not info or not info.get("exists"):
            rows.append(
                f"{html.escape(display)}: "
                f"<b style='color:{bad}'>missing</b>"
            )
            continue
        bits = []
        if info.get("readable") != "yes":
            bits.append(f"<span style='color:{bad}'>not readable</span>")
        perm = info.get("perm", "?")
        bits.append(
            f"perm {perm}"
            if perm in ("600", "400")
            else f"<span style='color:{warn}'>perm {perm}</span>"
        )
        bits.append("CONFLUENCE_TOKEN " + mark(info.get("op") == "yes"))
        if key == "secrets":
            bits.append(
                "TEST_CONFLUENCE_TOKEN " + mark(info.get("test") == "yes")
            )
        rows.append(
            f"{html.escape(display)}: "
            f"<b style='color:{ok}'>present</b> (" + ", ".join(bits) + ")"
        )
    return "<br>".join(rows)


class _ClusterSecretsCheckWorker(QtCore.QObject):
    """Check the two cluster secret files over SSH, off the GUI thread.

    Emits ``finished(ok, message, parsed)``: ``parsed`` is the
    :func:`_parse_cluster_check` dict on success, else ``None`` with the error
    in ``message``. No secret value is ever read back (``grep -q`` only).
    """

    finished = QtCore.pyqtSignal(bool, str, object)

    def __init__(self, communicator) -> None:
        super().__init__()
        self._comm = communicator

    def run(self) -> None:
        try:
            res = self._comm.execute_ssh_command(_cluster_check_command())
        except Exception as e:  # pragma: no cover - defensive
            self.finished.emit(False, f"SSH failed: {e}", None)
            return
        if not res.get("success"):
            self.finished.emit(
                False,
                (res.get("stderr") or "").strip() or "SSH command failed",
                None,
            )
            return
        self.finished.emit(
            True, "", _parse_cluster_check(res.get("stdout", "") or "")
        )


class _ClusterSecretsUploadWorker(QtCore.QObject):
    """Write the two secret files over SSH, off the GUI thread.

    Token values travel only inside the scp'd file bodies (chmod 600); they are
    never placed on an ssh/scp command line, so they stay out of logs and the
    remote process table. Emits ``finished(ok, message)``.
    """

    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, communicator, op_token, test_token) -> None:
        super().__init__()
        self._comm = communicator
        self._op_token = op_token
        self._test_token = test_token

    def run(self) -> None:
        try:
            ok, message = self._do_upload()
        except Exception as e:  # pragma: no cover - defensive
            ok, message = False, f"Upload failed: {e}"
        self.finished.emit(ok, message)

    def _do_upload(self):
        res = self._comm.execute_ssh_command('printf %s "$HOME"')
        home = (res.get("stdout") or "").strip()
        if not res.get("success") or not home:
            return False, "Could not resolve the remote home directory."
        env_dir = f"{home}/.config/picasso_workflow"
        env_path = f"{env_dir}/.env"
        secrets_path = f"{home}/.picasso_secrets"
        mk = self._comm.execute_ssh_command(
            f"mkdir -p {shlex.quote(env_dir)} "
            f"&& chmod 700 {shlex.quote(env_dir)}"
        )
        if not mk.get("success"):
            return False, (
                f"Could not create {env_dir}: "
                + ((mk.get("stderr") or "").strip())
            )
        # Stage each file inside the 700-mode env_dir, tighten it to 600, then
        # mv it into place. scp lands the body under the host umask (often
        # world-readable); staging under the 700 dir hides it until the mode is
        # fixed, and the mv publishes it atomically at 600 -- so the token file
        # is never world-readable at its real path (and a pre-existing 400 file
        # is still replaceable, since mv needs only the parent dir's write bit).
        for content, path, staging_name in (
            (_env_file_content(self._op_token), env_path, ".env.upload.tmp"),
            (
                _secrets_file_content(self._op_token, self._test_token),
                secrets_path,
                ".secrets.upload.tmp",
            ),
        ):
            staging = f"{env_dir}/{staging_name}"
            put = self._comm.put_file(content, staging)
            if not put.get("success"):
                return False, (
                    f"Could not write {path}: "
                    + ((put.get("stderr") or "").strip())
                )
            mv = self._comm.execute_ssh_command(
                f"chmod 600 {shlex.quote(staging)} "
                f"&& mv -f {shlex.quote(staging)} {shlex.quote(path)}"
            )
            if not mv.get("success"):
                # Best-effort cleanup so a failed upload leaves no stray temp.
                self._comm.execute_ssh_command(f"rm -f {shlex.quote(staging)}")
                return False, (
                    f"Could not install {path}: "
                    + ((mv.get("stderr") or "").strip())
                )
        return True, f"Wrote {env_path} and {secrets_path} (chmod 600)."


class DoctorTab(QtWidgets.QWidget):
    """Read-only diagnostics tab with opt-in connection tests."""

    _STATUS_COLORS = {
        "ok": "#2e7d32",
        "warn": "#b26a00",
        "fail": "#c62828",
    }

    def __init__(self, main_window=None) -> None:
        """Initialize the tab.

        Parameters
        ----------
        main_window : picasso_workflow.gui.Window, optional
            The main window (unused for now; kept for symmetry with the other
            tabs and future use).
        """
        super().__init__()
        self.main_window = main_window
        self._conn_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._conn_results: dict[str, QtWidgets.QLabel] = {}
        # live connection-test threads; stopped on app quit (destroying a
        # running QThread aborts the process)
        self._threads: list[QtCore.QThread] = []
        self._quit_hook_connected = False
        # True while a cluster-secrets check/upload worker is in flight, so a
        # second op (or a refresh) can't start a concurrent one.
        self._cluster_busy = False
        self._build_ui()
        self.refresh()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        # Build everything into a container, then drop that into a scroll area
        # (see the end of this method) so the whole tab scrolls when the window
        # is shorter than its content -- a small screen would otherwise cut off
        # the lower sections with no way to reach them.
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        top = QtWidgets.QHBoxLayout()
        intro = QtWidgets.QLabel(
            "Read-only diagnostics. Secrets are shown as presence only "
            "(never their value). 'Test connection' makes a live "
            "authenticated request."
        )
        intro.setWordWrap(True)
        top.addWidget(intro, 1)
        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.setToolTip(
            "Re-read the environment and configuration diagnostics."
        )
        refresh_button.clicked.connect(self.refresh)
        top.addWidget(refresh_button)
        layout.addLayout(top)

        self.info_view = QtWidgets.QTextBrowser()
        self.info_view.setOpenExternalLinks(False)
        # Keep the diagnostics box usefully tall even inside the outer scroll
        # area (it scrolls its own long content); the outer area owns overflow
        # of the sections below it.
        self.info_view.setMinimumHeight(220)
        layout.addWidget(self.info_view, 1)

        # --- connections: per-profile test buttons ---
        conn_group = QtWidgets.QGroupBox("Connection tests")
        conn_layout = QtWidgets.QGridLayout(conn_group)
        for row, profile in enumerate(_test_profiles()):
            conn_layout.addWidget(QtWidgets.QLabel(profile), row, 0)
            button = QtWidgets.QPushButton("Test connection")
            button.setToolTip(
                "Make one authenticated Confluence request for this profile "
                "(runs in the background)."
            )
            button.clicked.connect(
                lambda _=False, p=profile: self._test_connection(p)
            )
            conn_layout.addWidget(button, row, 1)
            result = QtWidgets.QLabel("not tested")
            result.setWordWrap(True)
            result.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            conn_layout.addWidget(result, row, 2)
            conn_layout.setColumnStretch(2, 1)
            self._conn_buttons[profile] = button
            self._conn_results[profile] = result
        layout.addWidget(conn_group)

        # --- cluster secrets: check / provision over SSH ---
        sec_group = QtWidgets.QGroupBox("Cluster secrets (SLURM)")
        sec_layout = QtWidgets.QGridLayout(sec_group)
        blurb = QtWidgets.QLabel(
            "Set up and check the Confluence token files on the cluster so "
            "containerized runs can report to Confluence -- no terminal "
            "needed. The token is written only into the files (chmod 600), "
            "never shown here or logged. Targets the login node selected "
            "below."
        )
        blurb.setWordWrap(True)
        sec_layout.addWidget(blurb, 0, 0, 1, 3)

        sec_layout.addWidget(QtWidgets.QLabel("Cluster:"), 1, 0)
        self._cluster_combo = QtWidgets.QComboBox()
        self._cluster_combo.setToolTip(
            "Which configured SLURM login node to target (independent of the "
            "Run tab)."
        )
        self._sync_cluster_combo(select=self._run_tab_host())
        self._cluster_combo.currentTextChanged.connect(
            lambda _=None: self._refresh_cluster_secrets_ui()
        )
        sec_layout.addWidget(self._cluster_combo, 1, 1)
        self._cluster_host_label = QtWidgets.QLabel("")
        sec_layout.addWidget(self._cluster_host_label, 1, 2)

        sec_layout.addWidget(
            QtWidgets.QLabel("Confluence token (operational):"), 2, 0
        )
        self._op_token_edit = QtWidgets.QLineEdit()
        self._op_token_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        sec_layout.addWidget(self._op_token_edit, 2, 1, 1, 2)

        sec_layout.addWidget(
            QtWidgets.QLabel("Confluence TEST token (optional):"), 3, 0
        )
        self._test_token_edit = QtWidgets.QLineEdit()
        self._test_token_edit.setEchoMode(
            QtWidgets.QLineEdit.EchoMode.Password
        )
        sec_layout.addWidget(self._test_token_edit, 3, 1, 1, 2)

        self._cluster_check_button = QtWidgets.QPushButton("Check cluster")
        self._cluster_check_button.setToolTip(
            "SSH to the cluster and report whether the two secret files "
            "exist, are readable, and hold the expected keys (never their "
            "values)."
        )
        self._cluster_check_button.clicked.connect(self._check_cluster_secrets)
        sec_layout.addWidget(self._cluster_check_button, 4, 0)

        self._cluster_upload_button = QtWidgets.QPushButton(
            "Upload to cluster…"
        )
        self._cluster_upload_button.setToolTip(
            "Write ~/.config/picasso_workflow/.env and ~/.picasso_secrets on "
            "the cluster (chmod 600) from the tokens above (or the ones "
            "configured locally)."
        )
        self._cluster_upload_button.clicked.connect(
            self._upload_cluster_secrets
        )
        sec_layout.addWidget(self._cluster_upload_button, 4, 1)

        self._cluster_ssh_button = QtWidgets.QPushButton(
            "Set up SSH connection…"
        )
        self._cluster_ssh_button.setToolTip(
            "Generate/select an SSH key and install it on the selected login "
            "node (opens the SSH setup dialog), so Check/Upload can connect."
        )
        self._cluster_ssh_button.clicked.connect(self._setup_cluster_ssh)
        sec_layout.addWidget(self._cluster_ssh_button, 4, 2)

        self._cluster_status = QtWidgets.QLabel("not checked")
        self._cluster_status.setWordWrap(True)
        self._cluster_status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._cluster_status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        sec_layout.addWidget(self._cluster_status, 5, 0, 1, 3)
        sec_layout.setColumnStretch(2, 1)
        layout.addWidget(sec_group)

        # Make the whole tab scrollable so no section is unreachable on a short
        # window. widgetResizable keeps the container at the viewport width (so
        # content wraps rather than scrolling horizontally); NoFrame avoids a
        # double border inside the tab.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(container)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # -- passive diagnostics -----------------------------------------------

    def refresh(self) -> None:
        """Recompute the diagnostics and (re)render the passive view."""
        self._refresh_cluster_secrets_ui()
        try:
            diag = collect_diagnostics()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Could not collect diagnostics: {e}")
            self.info_view.setPlainText(f"Could not collect diagnostics: {e}")
            return
        self.info_view.setHtml(self._render_html(diag))

    @staticmethod
    def _row(label: str, value) -> str:
        return (
            f"<tr><td style='padding-right:16px;color:#555'>"
            f"{html.escape(str(label))}</td>"
            f"<td>{html.escape('' if value is None else str(value))}</td></tr>"
        )

    def _render_html(self, diag: dict) -> str:
        parts: list[str] = ["<style>h3{margin:12px 0 4px}</style>"]

        v = diag["versions"]
        parts.append("<h3>Versions</h3><table>")
        parts.append(self._row("picasso-workflow", v["picasso_workflow"]))
        parts.append(self._row("picasso", v["picasso"]))
        parts.append(self._row("Python", v["python"]))
        parts.append(self._row("platform", v["platform"]))
        parts.append(
            self._row("on SLURM cluster", diag["cluster"]["on_cluster"])
        )
        parts.append("</table>")

        parts.append("<h3>Config layers (low &rarr; high priority)</h3>")
        sources = diag.get("config_sources") or []
        if sources:
            parts.append("<table>")
            for layer, path, loaded in sources:
                mark = "loaded" if loaded else "not found"
                parts.append(self._row(f"{layer} [{mark}]", path))
            parts.append("</table>")
        else:
            parts.append("<i>no config sources recorded</i>")

        parts.append("<h3>.env</h3><table>")

        def _keys_row(keys):
            if keys:
                parts.append(self._row("keys defined", ", ".join(keys)))

        # The per-user .env is the file the GUI writes and the package loads at
        # import; always show its expected path so the user knows where it
        # belongs, whether or not it exists yet.
        user_env = diag.get("user_env") or {}
        user_path = user_env.get("path")
        if user_path:
            status = "found" if user_env.get("exists") else "not found"
            parts.append(self._row("per-user .env", f"{user_path} [{status}]"))
            _keys_row(user_env.get("keys"))
        # find_dotenv() scans upward from the installed package dir; show it only
        # when it located a *different* .env (e.g. a project-local one), to avoid
        # a confusing duplicate row for the per-user file.
        dotenv = diag.get("dotenv") or {}
        scanned = dotenv.get("path")
        if scanned and scanned != user_path:
            parts.append(self._row("also loaded (scanned)", scanned))
            _keys_row(dotenv.get("keys"))
        elif not user_path:
            parts.append(self._row("path", scanned or "none found"))
            _keys_row(dotenv.get("keys"))
        parts.append("</table>")

        log = diag.get("logging") or {}
        parts.append("<h3>Logging</h3><table>")
        parts.append(self._row("logfile", log.get("logfile")))
        parts.append(
            self._row(
                "rotation / retention",
                f"{log.get('rotation')} / {log.get('retention')}",
            )
        )
        parts.append(
            self._row(
                "levels (file / stderr)",
                f"{log.get('file_level')} / {log.get('stderr_level')}",
            )
        )
        parts.append("</table>")

        parts.append("<h3>Confluence profiles</h3>")
        for profile, s in diag.get("confluence", {}).items():
            parts.append(f"<b>{html.escape(profile)}</b><table>")
            if "error" in s:
                parts.append(self._row("error", s["error"]))
            else:
                parts.append(self._row("URL", s.get("base_url")))
                parts.append(self._row("space", s.get("space_key")))
                parts.append(
                    self._row("DefaultPage", s.get("parent_page_title"))
                )
                parts.append(self._row("username", s.get("username")))
                parts.append(
                    self._row(
                        "token", "set" if s.get("token_set") else "not set"
                    )
                )
            parts.append("</table>")

        parts.append("<h3>Environment variables</h3><table>")
        for name, status in diag.get("env", {}).items():
            parts.append(self._row(name, status))
        parts.append("</table>")

        return "".join(parts)

    # -- connection tests (opt-in, background) -----------------------------

    def _test_connection(self, profile: str) -> None:
        """Run a Confluence connection test for ``profile`` off-thread."""
        self._conn_buttons[profile].setEnabled(False)
        result = self._conn_results[profile]
        result.setText("testing...")
        result.setStyleSheet("")

        worker = _ConnTestWorker(profile)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        # keep the worker alive for the thread's lifetime; otherwise PyQt may
        # garbage-collect it before ``started`` fires and the thread hangs
        thread._pw_worker = worker
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_conn_result)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda t=thread: self._forget_thread(t))
        self._threads.append(thread)
        self._connect_quit_hook()
        thread.start()

    def _on_conn_result(self, profile: str, status: str, message: str) -> None:
        label = self._conn_results[profile]
        label.setText(f"{status.upper()}: {message}")
        color = self._STATUS_COLORS.get(status, "")
        label.setStyleSheet(f"color: {color};" if color else "")
        self._conn_buttons[profile].setEnabled(True)

    # -- cluster secrets (check / provision, background) -------------------

    def _run_tab_host(self) -> str:
        """The cluster host selected in the Run tab (``""`` if unavailable)."""
        combo = getattr(self.main_window, "cluster_host_combo", None)
        if combo is None:
            return ""
        try:
            return str(combo.currentText()).strip()
        except Exception:  # pragma: no cover - defensive
            return ""

    def _cluster_host(self) -> str:
        """The login node this tab targets: the Doctor-tab selector, falling
        back to the Run tab's selection."""
        combo = getattr(self, "_cluster_combo", None)
        if combo is not None:
            text = str(combo.currentText()).strip()
            if text:
                return text
        return self._run_tab_host()

    def _sync_cluster_combo(self, select=None) -> None:
        """(Re)populate the cluster selector from ``SlurmLoginNodes``.

        Preserves the current selection, or selects ``select`` when given and
        present. Signals are blocked so this doesn't trigger a refresh.
        """
        combo = getattr(self, "_cluster_combo", None)
        if combo is None:
            return
        from picasso_workflow import CONFIG

        want = select or str(combo.currentText()).strip()
        combo.blockSignals(True)
        combo.clear()
        for host in CONFIG.get("SlurmLoginNodes") or {}:
            combo.addItem(host)
        idx = combo.findText(want) if want else -1
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _setup_cluster_ssh(self) -> None:
        """Open the SSH-setup dialog targeted at the selected login node."""
        mw = self.main_window
        if mw is None or not hasattr(mw, "on_configure_ssh"):
            return
        from picasso_workflow import CONFIG

        host = self._cluster_host()
        login_node = (CONFIG.get("SlurmLoginNodes") or {}).get(host, host)
        saved = mw.on_configure_ssh(default_host=login_node)
        # The dialog may have added/selected a (possibly new) node; re-sync and
        # select whatever was saved so Check/Upload target it, else keep the
        # current selection.
        self._sync_cluster_combo(select=saved or None)
        self._refresh_cluster_secrets_ui()

    def _refresh_cluster_secrets_ui(self) -> None:
        """Update the host label / placeholders / button-enabled state."""
        host = self._cluster_host()
        from picasso_workflow import CONFIG

        login_node = (CONFIG.get("SlurmLoginNodes") or {}).get(host, "")
        self._cluster_host_label.setText(
            f"→ {login_node}" if login_node else ""
        )
        can_ssh = bool(host) and hasattr(
            self.main_window, "_ensure_slurm_communicator"
        )
        # Disabled while a check/upload is in flight so the two can't overlap.
        enabled = can_ssh and not self._cluster_busy
        self._cluster_check_button.setEnabled(enabled)
        self._cluster_upload_button.setEnabled(enabled)
        # The SSH-setup button is usable even when SSH isn't working yet --
        # that's its whole point -- so it only needs a host + the main window.
        if hasattr(self, "_cluster_ssh_button"):
            self._cluster_ssh_button.setEnabled(
                bool(host) and hasattr(self.main_window, "on_configure_ssh")
            )
        op_local, test_local = _local_tokens()
        self._op_token_edit.setPlaceholderText(
            "leave blank to reuse the token configured locally"
            if op_local
            else "required (or configure it locally via onboarding / Help)"
        )
        self._test_token_edit.setPlaceholderText(
            "leave blank to reuse the local TEST token"
            if test_local
            else "optional"
        )

    def _set_cluster_busy(self, busy: bool) -> None:
        """Mark a cluster op in/out of flight and refresh button state."""
        self._cluster_busy = busy
        self._refresh_cluster_secrets_ui()

    def _start_cluster_worker(self, worker, on_done) -> None:
        """Run a cluster-secrets ``worker`` off-thread (shared lifecycle)."""
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread._pw_worker = worker
        thread.started.connect(worker.run)
        # Connect the bound method (a slot on this QObject, in the GUI
        # thread), NOT a lambda: a context-less functor connection would run
        # in the worker thread and touch widgets off the GUI thread.
        worker.finished.connect(on_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda t=thread: self._forget_thread(t))
        self._threads.append(thread)
        self._connect_quit_hook()
        thread.start()

    def _check_cluster_secrets(self) -> None:
        """Check the two secret files on the selected cluster, off-thread."""
        if self._cluster_busy:
            return
        host = self._cluster_host()
        if not host or not hasattr(
            self.main_window, "_ensure_slurm_communicator"
        ):
            return
        try:
            comm = self.main_window._ensure_slurm_communicator(host)
        except Exception as e:
            self._cluster_status.setText(
                f"<span style='color:#c62828'>Could not connect: "
                f"{html.escape(str(e))}</span>"
            )
            return
        self._set_cluster_busy(True)
        self._cluster_status.setText("checking…")
        self._start_cluster_worker(
            _ClusterSecretsCheckWorker(comm), self._on_cluster_check_result
        )

    def _on_cluster_check_result(self, ok, message, parsed) -> None:
        self._set_cluster_busy(False)
        if not ok:
            hint = (
                " &mdash; if the SSH key or host isn't set up yet, use "
                "<b>Set up SSH connection…</b> above."
            )
            self._cluster_status.setText(
                f"<span style='color:#c62828'>Check failed: "
                f"{html.escape(message)}</span>{hint}"
            )
            return
        self._cluster_status.setText(_format_cluster_check(parsed or {}))

    def _upload_cluster_secrets(self) -> None:
        """Write the two secret files to the selected cluster, off-thread."""
        if self._cluster_busy:
            return
        host = self._cluster_host()
        if not host or not hasattr(
            self.main_window, "_ensure_slurm_communicator"
        ):
            return
        op_local, test_local = _local_tokens()
        op = self._op_token_edit.text().strip() or (op_local or "")
        test = self._test_token_edit.text().strip() or (test_local or "")
        if not op:
            QtWidgets.QMessageBox.warning(
                self,
                "No operational token",
                "Enter the operational Confluence token above, or configure "
                "it locally first (onboarding / Help → save token), then "
                "retry.",
            )
            return
        keys = "CONFLUENCE_TOKEN" + (", TEST_CONFLUENCE_TOKEN" if test else "")
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Upload secrets to cluster",
            f"Write ~/.config/picasso_workflow/.env and ~/.picasso_secrets "
            f"on <b>{html.escape(host)}</b> with {keys}? Any existing files "
            f"are overwritten (chmod 600).",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            comm = self.main_window._ensure_slurm_communicator(host)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Could not connect", str(e))
            return
        self._set_cluster_busy(True)
        self._cluster_status.setText("uploading…")
        self._start_cluster_worker(
            _ClusterSecretsUploadWorker(comm, op, test),
            self._on_cluster_upload_result,
        )

    def _on_cluster_upload_result(self, ok, message) -> None:
        self._set_cluster_busy(False)
        # Don't leave token values sitting in the widgets after an upload.
        self._op_token_edit.clear()
        self._test_token_edit.clear()
        color = "#2e7d32" if ok else "#c62828"
        self._cluster_status.setText(
            f"<span style='color:{color}'>{html.escape(message)}</span>"
        )
        if ok:
            # Re-check so the status reflects what's now on the cluster.
            self._check_cluster_secrets()

    def _forget_thread(self, thread: QtCore.QThread) -> None:
        if thread in self._threads:
            self._threads.remove(thread)

    def _connect_quit_hook(self) -> None:
        """Stop in-flight test threads on application shutdown (once)."""
        if self._quit_hook_connected:
            return
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_threads)
            self._quit_hook_connected = True

    def _stop_threads(self) -> None:
        """Quit and join any running test threads (destroying a live QThread
        aborts the process), then drop them."""
        for thread in list(self._threads):
            thread.quit()
            thread.wait()
        self._threads.clear()
