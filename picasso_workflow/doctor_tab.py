#!/usr/bin/env python
"""The GUI "Doctor" tab: read-only diagnostics for the running GUI.

Surfaces, at a glance, the state that otherwise only lives in the logfile:
which ``config.yaml`` layers were loaded and from where, which ``.env`` was
picked up, where the log is written, the picasso-workflow / picasso / Python
versions, whether we are on a SLURM node, and -- per Confluence profile --
the resolved (non-secret) connection settings plus whether a token is set.

Everything shown is passive and gathered without side effects; **secrets are
never displayed as values**, only as "set" / "not set" (see ``_SECRET_ENV``).
The only active feature is an explicit *Test connection* button per Confluence
profile: it makes a single authenticated request on a background thread (so
the GUI never blocks) and reports OK / warning / failure. No network call is
made when the tab is merely opened or refreshed.

Author: Heinrich Grabmayr
"""

from __future__ import annotations

import html
import os
import platform
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


def collect_diagnostics() -> dict:
    """Gather the read-only diagnostics shown by the tab (no secrets)."""
    import picasso_workflow as pw

    diag = getattr(pw, "DIAGNOSTICS", {}) or {}
    return {
        "versions": _versions(),
        "cluster": {"on_cluster": getattr(pw, "ON_CLUSTER", None)},
        "config_sources": diag.get("config_sources", []),
        "dotenv": diag.get("dotenv", {}),
        "logging": diag.get("logging", {}),
        "env": {name: _env_status(name) for name in _ENV_VARS},
        "confluence": {p: _confluence_summary(p) for p in _PROFILES},
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
        self._build_ui()
        self.refresh()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

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
        layout.addWidget(self.info_view, 1)

        # --- connections: per-profile test buttons ---
        conn_group = QtWidgets.QGroupBox("Connection tests")
        conn_layout = QtWidgets.QGridLayout(conn_group)
        for row, profile in enumerate(_PROFILES):
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

    # -- passive diagnostics -----------------------------------------------

    def refresh(self) -> None:
        """Recompute the diagnostics and (re)render the passive view."""
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

        dotenv = diag.get("dotenv") or {}
        parts.append("<h3>.env</h3><table>")
        parts.append(self._row("path", dotenv.get("path") or "none found"))
        if dotenv.get("keys"):
            parts.append(self._row("keys defined", ", ".join(dotenv["keys"])))
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
