"""Headless engine for onboarding a user's SSH link to a SLURM login node.

Mirrors monet's ``setup_core.py``: state detection, the SSH connection probe,
SSH-key discovery/generation and per-user ``config.yaml`` persistence live here
with NO Qt import, so they unit-test without a display. The thin PyQt dialog in
``gui.py`` (``SshSetupDialog``) wraps these.

Transport is the repo's subprocess-based
:class:`picasso_workflow.slurm_submit.SlurmCommunicator` -- no paramiko/fabric.
The public key is installed on the login node by the user running
``ssh-copy-id`` once (a password prompt the OS handles, not this app); this
module locates/creates the key, emits the exact command, and verifies with a
real connection probe before anything is persisted ("test before save").
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from picasso_workflow.slurm_submit import SlurmCommunicator

# SSH probe outcome categories.
CAT_OK = "ok"
CAT_AUTH = "auth"  # reachable, but the key was not accepted yet
CAT_UNKNOWN_HOST = "unknown_host"
CAT_TIMEOUT = "timeout"
CAT_KEY_PERMS = "key_perms"  # local private-key permissions too open
CAT_NO_KEY = "no_key"  # the configured key path does not exist
CAT_ERROR = "error"


@dataclass
class SshVerdict:
    """Outcome of :func:`test_ssh` -- a clear, user-facing verdict."""

    ok: bool
    category: str
    message: str
    detail: str = ""


@dataclass
class ClusterState:
    """What is already configured for cluster SSH (from CONFIG + ~/.ssh)."""

    hosts: dict = field(default_factory=dict)  # {label: login-node hostname}
    usernames: dict = field(default_factory=dict)  # {label: username}
    key_path: str | None = None
    has_key: bool = False


def resolve_username(username):
    """Expand a literal ``$USER`` (or empty) to the local login name.

    Matches ``gui._ensure_slurm_communicator``, which expands ``$USER`` with
    :func:`getpass.getuser` before building the connection -- so a probe and a
    later submission authenticate as the same real user.
    """
    name = (username or "").strip()
    if not name or name == "$USER":
        return getpass.getuser()
    return name


def _as_port(port):
    """Coerce ``port`` to an int, defaulting to 22 on anything non-numeric."""
    try:
        return int(port)
    except (TypeError, ValueError):
        return 22


# --------------------------------------------------------------------------- #
# SSH connection probe
# --------------------------------------------------------------------------- #
def test_ssh(
    hostname,
    username,
    port=22,
    ssh_key_path=None,
    timeout=15,
    *,
    comm_factory=None,
):
    """Probe SSH key-auth to ``username@hostname`` and classify the outcome.

    Uses the same subprocess ``ssh`` transport the submit path uses, so a
    passing probe here means a passing submission later. ``comm_factory`` is
    injectable for tests; it defaults to
    :class:`~picasso_workflow.slurm_submit.SlurmCommunicator`.
    """
    comm_factory = comm_factory or SlurmCommunicator
    if not hostname or not username:
        return SshVerdict(
            False, CAT_ERROR, "A host and a username are both required."
        )
    if ssh_key_path and not os.path.exists(ssh_key_path):
        return SshVerdict(
            False,
            CAT_NO_KEY,
            f"SSH key not found: {ssh_key_path}. "
            "Generate a new key or select an existing one.",
        )
    comm = comm_factory(
        hostname,
        username,
        port=port,
        ssh_key_path=ssh_key_path,
        timeout=timeout,
    )
    try:
        result = comm.execute_ssh_command(
            "echo picasso-ssh-ok", batch_mode=True
        )
    except FileNotFoundError:
        return SshVerdict(
            False,
            CAT_ERROR,
            "The 'ssh' command was not found on this machine. Install an "
            "OpenSSH client first.",
        )
    except Exception as exc:  # subprocess / unexpected transport error
        return _classify("", str(exc), hostname, ssh_key_path)
    stdout = result.get("stdout") or ""
    if result.get("success") and "picasso-ssh-ok" in stdout:
        return SshVerdict(
            True,
            CAT_OK,
            f"Connected to {username}@{hostname} and authenticated "
            "with the SSH key.",
        )
    return _classify(
        stdout, result.get("stderr") or "", hostname, ssh_key_path
    )


def _classify(stdout, stderr, hostname, ssh_key_path):
    """Map an ``ssh`` failure (stderr) to a specific, actionable verdict."""
    s = (stderr or "").lower()
    if (
        "could not resolve hostname" in s
        or "name or service not known" in s
        or "nodename nor servname" in s
        or "name does not resolve" in s
    ):
        return SshVerdict(
            False,
            CAT_UNKNOWN_HOST,
            f"Host '{hostname}' could not be resolved. Check the login-node "
            "name, and that you are on the institute network / VPN.",
            stderr,
        )
    if "timed out" in s or "timeout" in s:
        return SshVerdict(
            False,
            CAT_TIMEOUT,
            f"The connection to {hostname} timed out. Check the host and port "
            "and that you can reach it (VPN?).",
            stderr,
        )
    if "permissions" in s and (
        "too open" in s or "private key" in s or "ignore key" in s
    ):
        return SshVerdict(
            False,
            CAT_KEY_PERMS,
            f"The private key {ssh_key_path} has permissions that are too "
            f"open. Run:  chmod 600 {ssh_key_path}",
            stderr,
        )
    if (
        "permission denied" in s
        or "publickey" in s
        or "authentication failed" in s
        or "too many authentication failures" in s
    ):
        return SshVerdict(
            False,
            CAT_AUTH,
            f"Reached {hostname}, but the key was not accepted. Install your "
            "public key on the login node (the ssh-copy-id step below), then "
            "test again.",
            stderr,
        )
    msg = (stderr or stdout or "SSH connection failed.").strip()
    return SshVerdict(False, CAT_ERROR, f"SSH failed: {msg}", stderr)


# --------------------------------------------------------------------------- #
# SSH-key discovery / generation
# --------------------------------------------------------------------------- #
def ssh_dir(home=None):
    """Return the ``~/.ssh`` directory (``home`` overridable for tests)."""
    base = Path(home) if home else Path.home()
    return base / ".ssh"


def discover_ssh_key(home=None):
    """Return an existing private-key path, or ``None``.

    Order (``id_rsa`` then ``id_ed25519``) mirrors
    ``gui._ensure_slurm_communicator`` so onboarding picks the same key the
    submit path will use.
    """
    d = ssh_dir(home)
    for name in ("id_rsa", "id_ed25519"):
        p = d / name
        if p.exists():
            return str(p)
    return None


def default_new_key_path(home=None):
    """Path for a freshly generated key (modern ed25519)."""
    return str(ssh_dir(home) / "id_ed25519")


def public_key_path(private_key_path):
    """The ``.pub`` sibling of a private-key path."""
    return str(private_key_path) + ".pub"


def generate_ssh_key(
    key_path=None,
    home=None,
    comment="picasso-workflow",
    *,
    runner=None,
):
    """Create an ed25519 key pair at ``key_path`` (no passphrase).

    No passphrase so the subprocess ``ssh`` the submit path uses is
    non-interactive. Returns the private-key path. Raises ``FileExistsError``
    if a key is already there (the caller decides whether to reuse it) and
    ``RuntimeError`` if ``ssh-keygen`` fails. ``runner`` is injectable (tests).
    """
    runner = runner or subprocess.run
    key_path = key_path or default_new_key_path(home)
    d = os.path.dirname(key_path)
    if d:
        os.makedirs(d, exist_ok=True)
        try:
            os.chmod(d, 0o700)
        except OSError:  # pragma: no cover - platform dependent
            pass
    if os.path.exists(key_path):
        raise FileExistsError(key_path)
    proc = runner(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            comment,
            "-f",
            key_path,
        ],
        capture_output=True,
        text=True,
    )
    if getattr(proc, "returncode", 1) != 0:
        raise RuntimeError(
            "ssh-keygen failed: " + (getattr(proc, "stderr", "") or "")
        )
    logger.info(f"Generated SSH key at {key_path}")
    return key_path


def ssh_copy_id_command(hostname, username, port=22, key_path=None):
    """The exact one-time command to install the public key on the node."""
    parts = ["ssh-copy-id"]
    if _as_port(port) != 22:
        parts += ["-p", str(_as_port(port))]
    if key_path:
        parts += ["-i", public_key_path(key_path)]
    parts.append(f"{username}@{hostname}")
    return " ".join(parts)


def install_key_command(
    hostname, username, port=22, key_path=None, platform=None
):
    """The one-time command to install the public key on the login node.

    POSIX uses ``ssh-copy-id``; **Windows** uses a ``type <pub> | ssh …`` pipe
    instead, because ``ssh-copy-id`` is not part of the Windows OpenSSH client.
    Either way the user runs it once in a terminal (which handles the password
    prompt). ``platform`` overrides ``sys.platform`` for tests.
    """
    plat = sys.platform if platform is None else platform
    if plat == "win32":
        pub = (
            public_key_path(key_path)
            if key_path
            else "%USERPROFILE%\\.ssh\\id_ed25519.pub"
        )
        p = f"-p {_as_port(port)} " if _as_port(port) != 22 else ""
        return (
            f'type "{pub}" | ssh {p}{username}@{hostname} '
            '"mkdir -p ~/.ssh && chmod 700 ~/.ssh && '
            'cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"'
        )
    return ssh_copy_id_command(
        hostname, username, port=port, key_path=key_path
    )


# --------------------------------------------------------------------------- #
# State detection + per-user config persistence
# --------------------------------------------------------------------------- #
def detect_state(config=None, home=None):
    """Summarise what is already configured for cluster SSH."""
    if config is None:
        from picasso_workflow import CONFIG as config
    hosts = dict(config.get("SlurmLoginNodes") or {})
    unames = config.get("LoginNodeUserNames")
    usernames = {}
    for label in hosts:
        if isinstance(unames, dict):
            usernames[label] = unames.get(label, "")
        elif isinstance(unames, str):
            usernames[label] = unames
        else:
            usernames[label] = ""
    key = discover_ssh_key(home)
    return ClusterState(
        hosts=hosts,
        usernames=usernames,
        key_path=key,
        has_key=key is not None,
    )


def _user_config_path():
    from picasso_workflow import _user_config_dir

    return _user_config_dir() / "config.yaml"


def write_user_config(updates, *, config_path=None):
    """Deep-merge ``updates`` into the per-user ``config.yaml`` (backup first).

    ``updates`` is a nested dict (e.g.
    ``{"SlurmLoginNodes": {label: hostname}}``). An existing file is copied to
    ``<name>.bak`` before writing, and the write is atomic (tmp + ``os.replace``).
    Note: re-serialising via YAML drops comments from the per-user file; the
    ``.bak`` preserves the original. Returns the path written.
    """
    import yaml

    from picasso_workflow import _deep_merge

    path = Path(config_path) if config_path else _user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                existing = yaml.safe_load(fh) or {}
        except Exception:  # pragma: no cover - defensive against a bad file
            existing = {}
        shutil.copy2(path, str(path) + ".bak")
    merged = _deep_merge(existing, updates)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(merged, fh, default_flow_style=False, sort_keys=False)
    os.replace(tmp, path)
    logger.info(f"Wrote cluster SSH settings to {path}")
    return str(path)


def reload():
    """Re-read the config layers and update the module-level CONFIG in place."""
    from picasso_workflow import reload_config

    return reload_config()


def persist_cluster_ssh(label, hostname, username, *, config_path=None):
    """Persist one login node (label->hostname + username) and reload CONFIG.

    ``label`` is the key shown in the Run-tab host combo; ``hostname`` is the
    actual SSH host. A caller onboarding a single node typically passes the
    hostname as both. Returns the config path written.
    """
    updates = {
        "SlurmLoginNodes": {label: hostname},
        "LoginNodeUserNames": {label: username},
    }
    path = write_user_config(updates, config_path=config_path)
    reload()
    return path
