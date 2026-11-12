"""GUI-free admin operations for picasso-workflow's cluster surface.

The admin console inspects the shared-mount Apptainer images, the SLURM queue,
and disk usage on the cluster login node. Unlike monet/picasso-registry (which
drive a Docker-deployed server over paramiko), picasso-workflow already has an
SSH transport -- :class:`picasso_workflow.slurm_submit.SlurmCommunicator`
(subprocess ``ssh``) -- so this core reuses it rather than adding a new
dependency. Connection + image settings come from ``CONFIG`` (``SlurmLoginNodes``
/ ``ClusterApptainer``), the same source the Run tab uses.

The command-output parsers (:func:`parse_image_listing`, :func:`parse_squeue`,
:func:`parse_sacct`, :func:`parse_df`) are pure functions kept apart from the
SSH calls so they unit-test without a cluster (mirrors monet's ``parse_status``
/ ``parse_token_list``).
"""

from __future__ import annotations

import getpass
import os
import shlex
from dataclasses import dataclass

from picasso_workflow import CONFIG
from picasso_workflow.slurm_submit import (
    SlurmCommunicator,
    apptainer_config,
    default_image_basename,
)


# ---------------------------------------------------------------------------
# Version / image naming
# ---------------------------------------------------------------------------
def default_test_image_basename(version):
    """Return the ``-test-`` ``.sif`` filename matching ``version``.

    The test image (built with the ``[dev]`` extra so it carries pytest) is
    named ``picasso-workflow-test-<version>.sif`` alongside the lean
    ``picasso-workflow-<version>.sif``. Returns ``None`` for the same
    unknown/dev versions :func:`slurm_submit.default_image_basename` rejects.
    """
    base = default_image_basename(version)
    if base is None:
        return None
    return base.replace("picasso-workflow-", "picasso-workflow-test-", 1)


def this_gui_version():
    """The installed ``picasso_workflow`` version, or ``"unknown"``.

    Prefers the dist metadata (present even in a non-git wheel install) and
    falls back to the importable ``__version__``.
    """
    try:
        import importlib.metadata

        return importlib.metadata.version("picasso_workflow")
    except Exception:
        try:
            from picasso_workflow import __version__

            return __version__
        except Exception:
            return "unknown"


def parse_image_name(name):
    """Split a ``.sif`` basename into ``(version, is_test)``.

    ``picasso-workflow-0.9.0.sif`` -> ``("0.9.0", False)``;
    ``picasso-workflow-test-0.9.0.sif`` -> ``("0.9.0", True)``;
    anything else (e.g. ``...-develop-<sha>.sif``) -> ``(None, is_test)`` with
    ``is_test`` reflecting the ``-test-`` marker.
    """
    stem = name[:-4] if name.endswith(".sif") else name
    is_test = stem.startswith("picasso-workflow-test-")
    if is_test:
        version = stem[len("picasso-workflow-test-") :]
    elif stem.startswith("picasso-workflow-"):
        version = stem[len("picasso-workflow-") :]
    else:
        return None, is_test
    # A plain release version is digits/dots; develop-<sha> and the like are
    # left unparsed (shown verbatim by the caller, not matched to a version).
    if version and all(c.isdigit() or c == "." for c in version):
        return version, is_test
    return None, is_test


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ImageInfo:
    """One ``.sif`` on the shared mount."""

    name: str
    version: str | None
    is_test: bool
    size: str  # human-readable (e.g. "4.2G"), "" if unknown
    mtime: str  # "YYYY-MM-DD HH:MM", "" if unknown


@dataclass
class JobInfo:
    """One SLURM job (from squeue or sacct)."""

    job_id: str
    name: str
    state: str
    elapsed: str
    extra: str = ""  # partition/reason (squeue) or end time (sacct)


@dataclass
class DiskUsage:
    """Disk usage for a path (from df)."""

    path: str
    size: str
    used: str
    avail: str
    use_pct: str


# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------
def parse_image_listing(stdout):
    """Parse ``ls -l --time-style=long-iso <dir>`` into ``[ImageInfo]``.

    Only regular files (lines starting ``-``) ending in ``.sif`` are kept.
    Fields: perms links owner group size YYYY-MM-DD HH:MM name...
    """
    images = []
    for line in stdout.splitlines():
        line = line.rstrip("\n")
        if not line or not line.startswith("-"):
            continue  # skip 'total N', directories, symlinks, blanks
        parts = line.split()
        if len(parts) < 8:
            continue
        name = " ".join(parts[7:])
        if not name.endswith(".sif"):
            continue
        version, is_test = parse_image_name(name)
        images.append(
            ImageInfo(
                name=name,
                version=version,
                is_test=is_test,
                size=_human_size(parts[4]),
                mtime=f"{parts[5]} {parts[6]}",
            )
        )
    images.sort(key=lambda i: i.name)
    return images


def _human_size(num_bytes):
    """Bytes (as a string) -> human-readable, best effort."""
    try:
        n = float(num_bytes)
    except (TypeError, ValueError):
        return str(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def parse_squeue(stdout):
    """Parse ``squeue -h -o '%i|%j|%T|%M|%P|%R'`` into ``[JobInfo]``."""
    jobs = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split("|")
        if len(f) < 4:
            continue
        extra = " ".join(x for x in (f[4:5] + f[5:6]) if x)
        jobs.append(
            JobInfo(
                job_id=f[0],
                name=f[1],
                state=f[2],
                elapsed=f[3],
                extra=extra,
            )
        )
    return jobs


def parse_sacct(stdout):
    """Parse ``sacct -X -n -P -o JobID,JobName,State,Elapsed,End``.

    ``-P`` emits pipe-separated fields, ``-n`` drops the header, ``-X`` keeps
    only the main job (not the ``.batch``/``.extern`` steps).
    """
    jobs = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split("|")
        if len(f) < 4:
            continue
        jobs.append(
            JobInfo(
                job_id=f[0],
                name=f[1],
                state=f[2],
                elapsed=f[3],
                extra=f[4] if len(f) > 4 else "",
            )
        )
    return jobs


def parse_df(stdout):
    """Parse ``df -h <path>`` (header + one data line) into ``DiskUsage``.

    Returns ``None`` when the output has no data line.
    """
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    f = lines[1].split()
    # Filesystem Size Used Avail Use% Mounted-on
    if len(f) < 6:
        return None
    return DiskUsage(
        path=" ".join(f[5:]),
        size=f[1],
        used=f[2],
        avail=f[3],
        use_pct=f[4],
    )


# ---------------------------------------------------------------------------
# Connection helpers (reused from the GUI's resolution)
# ---------------------------------------------------------------------------
def discover_ssh_key():
    """Prefer ``~/.ssh/id_rsa``, else ``id_ed25519``, else the last candidate.

    Mirrors ``gui.Window._ensure_slurm_communicator`` so the admin console
    authenticates exactly like the Run tab.
    """
    candidate = None
    for name in ("id_rsa", "id_ed25519"):
        candidate = os.path.join(os.path.expanduser("~"), ".ssh", name)
        if os.path.exists(candidate):
            break
    return candidate


def resolve_login_node(host):
    """Map a cluster key to its login-node hostname via ``SlurmLoginNodes``.

    An unknown key is used verbatim (so a hand-typed hostname still works).
    """
    return CONFIG.get("SlurmLoginNodes", {}).get(str(host), str(host))


def cluster_hosts():
    """Configured cluster keys (``SlurmLoginNodes`` keys), for the host picker."""
    return list(CONFIG.get("SlurmLoginNodes", {}).keys())


def build_admin(host, username=None, ssh_key_path=None):
    """Build a :class:`ClusterAdmin` for ``host`` from the current settings."""
    login_node = resolve_login_node(host)
    if not username or username == "$USER":
        username = getpass.getuser()
    if ssh_key_path is None:
        ssh_key_path = discover_ssh_key()
    comm = SlurmCommunicator(
        login_node, username, port=22, ssh_key_path=ssh_key_path
    )
    return ClusterAdmin(comm, host, username)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------
class ClusterAdmin:
    """Thin SSH operations layer over a :class:`SlurmCommunicator`.

    One instance per operation is built from a snapshot of the connection
    fields (so the GUI worker never touches Qt widgets off-thread); the
    communicator opens no persistent connection -- each call is a fresh
    subprocess ``ssh``.
    """

    def __init__(self, comm, host, username):
        self.comm = comm
        self.host = host
        self.username = username

    def _run(self, command):
        """Run ``command`` over SSH; raise ``RuntimeError`` on failure."""
        result = self.comm.execute_ssh_command(command)
        if not result.get("success"):
            raise RuntimeError(
                f"{command!r} failed (exit {result.get('return_code')}): "
                f"{(result.get('stderr', '') or '').strip()}"
            )
        return result.get("stdout", "") or ""

    def test_connection(self):
        """True if a trivial SSH command succeeds."""
        return self.comm.test_connection()

    def list_images(self):
        """Inventory of ``.sif`` images in the configured ``image_dir``."""
        image_dir, _module, _binds = apptainer_config(self.host)
        if not image_dir:
            raise RuntimeError(
                "ClusterApptainer.image_dir is not set in config.yaml."
            )
        out = self._run(
            f"ls -l --time-style=long-iso {shlex.quote(image_dir)}"
        )
        return parse_image_listing(out)

    def image_dir(self):
        """The configured shared-image directory for this host (or "")."""
        image_dir, _module, _binds = apptainer_config(self.host)
        return image_dir

    def queue(self):
        """The user's current SLURM queue (squeue)."""
        out = self._run(
            "squeue -h -u "
            + shlex.quote(self.username)
            + " -o '%i|%j|%T|%M|%P|%R'"
        )
        return parse_squeue(out)

    def recent_jobs(self, days=3):
        """The user's recent finished/running jobs (sacct)."""
        out = self._run(
            "sacct -X -n -P "
            + "-u "
            + shlex.quote(self.username)
            + f" -S now-{int(days)}days "
            + "-o JobID,JobName,State,Elapsed,End"
        )
        return parse_sacct(out)

    def disk_usage(self):
        """``df -h`` for the image directory (or ``None`` if unset)."""
        image_dir, _module, _binds = apptainer_config(self.host)
        if not image_dir:
            return None
        out = self._run(f"df -h {shlex.quote(image_dir)}")
        return parse_df(out)
