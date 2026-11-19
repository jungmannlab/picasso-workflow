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

from loguru import logger

from picasso_workflow import CONFIG
from picasso_workflow.slurm_submit import (
    SlurmCommunicator,
    apptainer_config,
    apptainer_test_results_dir,
    default_image_basename,
    resolve_pw_version,
)

#: Field separator for ``squeue``/``sacct`` ``-o`` output. A tab never appears
#: in a SLURM job name / reason, so splitting on it (rather than ``|``) is
#: robust to a pipe in a user's job name.
_SEP = "\t"


# ---------------------------------------------------------------------------
# Version / image naming
# ---------------------------------------------------------------------------
def this_gui_version():
    """The installed ``picasso_workflow`` version, or ``"unknown"``.

    Delegates to :func:`slurm_submit.resolve_pw_version` so the admin console
    and the Run tab's image auto-match report the SAME version (the Images
    tab's "this GUI" highlighting depends on the two agreeing).
    """
    return resolve_pw_version()


def default_test_image_basename(version):
    """Return the ``-test-`` ``.sif`` filename matching ``version``.

    The test image (built with the ``[dev]`` extra so it carries pytest) is
    named ``picasso-workflow-test-<version>.sif`` alongside the lean
    ``picasso-workflow-<version>.sif``; the Test-runs tab defaults to it.
    Returns ``None`` for the unknown/dev versions
    :func:`slurm_submit.default_image_basename` rejects.
    """
    base = default_image_basename(version)
    if base is None:
        return None
    return base.replace("picasso-workflow-", "picasso-workflow-test-", 1)


def _split_image_name(name):
    """``(label, is_test, matched)`` for a ``.sif`` basename.

    ``label`` is the stem after the ``picasso-workflow[-test-]`` prefix (or the
    whole stem when no prefix matched); ``is_test`` reflects the ``-test-``
    marker; ``matched`` says a picasso-workflow prefix was present. Single
    source of the prefix-stripping shared by :func:`parse_image_name` and
    :func:`image_tag`.
    """
    stem = name[:-4] if name.endswith(".sif") else name
    if stem.startswith("picasso-workflow-test-"):
        return stem[len("picasso-workflow-test-") :], True, True
    if stem.startswith("picasso-workflow-"):
        return stem[len("picasso-workflow-") :], False, True
    return stem, False, False


def parse_image_name(name):
    """Split a ``.sif`` basename into ``(version, is_test)``.

    ``picasso-workflow-0.9.0.sif`` -> ``("0.9.0", False)``;
    ``picasso-workflow-test-0.9.0.sif`` -> ``("0.9.0", True)``;
    anything else (e.g. ``...-develop-<sha>.sif``) -> ``(None, is_test)`` with
    ``is_test`` reflecting the ``-test-`` marker.
    """
    label, is_test, matched = _split_image_name(name)
    if not matched:
        return None, is_test
    # A plain release version is dot-separated digit groups ("0.9.0");
    # develop-<sha>, malformed dotted names ("0..0"), etc. are left unparsed
    # (shown verbatim by the caller, not matched to a version).
    groups = label.split(".")
    if label and all(g.isdigit() for g in groups):
        return label, is_test
    return None, is_test


def configured_test_data_dir(config=None):
    """The configured test-data dir (``TestData.directory``), or ``""``.

    The same source ``tools/cluster_tests/submit_all.sh`` and the
    ``network_test_data`` pytest fixture use to resolve ``PW_TEST_DATA_DIR``
    when the env var is unset -- so a GUI-launched test run exercises the same
    datasets (tier 4 + the per-template workflow runs). ``config`` defaults to
    the package ``CONFIG``. Note it must be valid on the cluster compute node
    (where ``submit_all.sh`` runs it is, being read from the login node's
    config).
    """
    if config is None:
        config = CONFIG
    # Tolerate a malformed (non-dict) TestData in a hand-edited config.yaml:
    # fall back to blank rather than crashing GUI startup on `.get`.
    test_data = config.get("TestData")
    if not isinstance(test_data, dict):
        return ""
    return str(test_data.get("directory") or "")


def image_tag(name):
    """The human-readable label after the prefix (for the Images table).

    ``picasso-workflow-0.9.0.sif`` -> ``"0.9.0"``;
    ``picasso-workflow-test-0.9.0.sif`` -> ``"0.9.0"``;
    ``picasso-workflow-develop-ef8b42.sif`` -> ``"develop-ef8b42"``. Unlike
    :func:`parse_image_name` (which returns a version only for a clean release,
    for match-highlighting), this always yields something to show.
    """
    return _split_image_name(name)[0]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ImageInfo:
    """One ``.sif`` on the shared mount."""

    name: str
    version: str | None  # clean release version (for match), else None
    tag: str  # label to display (release version, or "develop-<sha>", …)
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


@dataclass
class DayStat:
    """Per-day job counts, for the Jobs & usage stats graph."""

    date: str  # "YYYY-MM-DD"
    total: int  # finished jobs that ended on this day (passed + failed)
    passed: int  # COMPLETED
    failed: int  # FAILED / CANCELLED / TIMEOUT / ...

    @property
    def pass_fraction(self):
        """Fraction of finished jobs that passed (0..1), or ``None`` if none."""
        return (self.passed / self.total) if self.total else None


# ---------------------------------------------------------------------------
# Job-state categorisation + over-time aggregation (pure, for the GUI)
# ---------------------------------------------------------------------------
#: Map a SLURM state word to a coarse phase the console groups jobs by. A state
#: can carry a suffix (e.g. ``CANCELLED by 1234``), so callers match the first
#: word. Anything unrecognised falls back to ``"other"``.
_PHASE_BY_STATE = {
    "PENDING": "queued",
    "CONFIGURING": "queued",
    "REQUEUED": "queued",
    "RUNNING": "running",
    "COMPLETING": "running",
    "RESIZING": "running",
    "SUSPENDED": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "failed",
    "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed",
    "NODE_FAIL": "failed",
    "BOOT_FAIL": "failed",
    "DEADLINE": "failed",
    "PREEMPTED": "failed",
}

#: Coarse phases, in the order the console reports them.
JOB_PHASES = ("queued", "running", "completed", "failed", "other")


def job_phase(state):
    """Coarse phase (:data:`JOB_PHASES`) for a SLURM ``state`` string.

    The first whitespace-separated word is matched (SLURM appends e.g.
    ``CANCELLED by <uid>``); an empty or unknown state is ``"other"``.
    """
    word = (state or "").split()[0].upper() if (state or "").strip() else ""
    return _PHASE_BY_STATE.get(word, "other")


def summarize_phases(jobs):
    """Count ``jobs`` (``[JobInfo]``) per :data:`JOB_PHASES`.

    Returns a dict with every phase key present (zero when none), so callers
    can format a stable "X queued · Y running · …" summary.
    """
    counts = {phase: 0 for phase in JOB_PHASES}
    for job in jobs:
        counts[job_phase(job.state)] += 1
    return counts


def sacct_end_date(end):
    """Return the ``YYYY-MM-DD`` day of a ``sacct`` End field, or ``None``.

    ``sacct`` End is ``YYYY-MM-DDTHH:MM:SS`` for a finished job, or a sentinel
    (``Unknown``, ``None``, blank) for one still queued/running — those yield
    ``None`` (not counted in the over-time stats).
    """
    s = (end or "").strip()
    if not s or s in ("Unknown", "None"):
        return None
    day = s.split("T", 1)[0]
    parts = day.split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return day
    return None


def aggregate_jobs_by_day(jobs):
    """Aggregate finished ``jobs`` into per-day :class:`DayStat` (date-sorted).

    Only jobs that actually finished (a parseable End date and a
    completed/failed phase) are counted; still queued/running jobs and
    ``.batch``/other phases are skipped. One row per day the user had finished
    jobs — the series the Jobs & usage stats graph plots.
    """
    by_day = {}
    for job in jobs:
        day = sacct_end_date(job.extra)
        if day is None:
            continue
        phase = job_phase(job.state)
        if phase not in ("completed", "failed"):
            continue
        passed, failed = by_day.get(day, (0, 0))
        if phase == "completed":
            passed += 1
        else:
            failed += 1
        by_day[day] = (passed, failed)
    return [
        DayStat(date=day, total=p + f, passed=p, failed=f)
        for day, (p, f) in sorted(by_day.items())
    ]


# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------
def parse_image_listing(stdout):
    """Parse ``ls -l --time-style=long-iso <dir>`` into ``[ImageInfo]``.

    Regular files (lines starting ``-``) and symlinks (``l``, e.g. a
    ``...-latest.sif`` alias) ending in ``.sif`` are kept. Fields: perms links
    owner group size YYYY-MM-DD HH:MM name[ -> target].
    """
    images = []
    for line in stdout.splitlines():
        line = line.rstrip("\n")
        if not line or line[0] not in "-l":
            continue  # skip 'total N', directories, blanks
        parts = line.split()
        if len(parts) < 8:
            continue
        name = " ".join(parts[7:])
        # Symlinks render as "name -> target"; keep the link's own name.
        name = name.split(" -> ", 1)[0]
        if not name.endswith(".sif"):
            continue
        version, is_test = parse_image_name(name)
        images.append(
            ImageInfo(
                name=name,
                version=version,
                tag=image_tag(name),
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
    """Parse tab-separated ``squeue -h -o`` output into ``[JobInfo]``.

    Columns: JobID, Name, State, Time, Partition, Reason (see
    :meth:`ClusterAdmin.queue`). Split on tab (``_SEP``) so a pipe in a job
    name cannot shift fields.
    """
    jobs = []
    for line in stdout.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        f = line.split(_SEP)
        if len(f) < 4:
            continue
        extra = " ".join(x for x in f[4:] if x)  # partition + reason
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
    """Parse tab-separated ``sacct`` output into ``[JobInfo]``.

    Columns: JobID, JobName, State, Elapsed, End (see
    :meth:`ClusterAdmin.recent_jobs`). ``-n`` drops the header, ``-X`` keeps
    only the main job (not the ``.batch``/``.extern`` steps); the tab delimiter
    is robust to a pipe in a job name.
    """
    jobs = []
    for line in stdout.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        f = line.split(_SEP)
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


def parse_jobs_tsv(text):
    """Parse a run's ``jobs.tsv`` manifest into ``(job_ids, templates)``.

    Each line is ``kind<TAB>jobid<TAB>label<TAB>dir`` (see
    :func:`testrun.render_jobs_tsv`): ``kind`` is ``tier`` or ``template``.
    Returns the de-duplicated job ids in first-seen order (a tier job appears
    on two rows) and the template runs as ``(dir, jobid)`` pairs (matching the
    GUI's ``_test_run_templates``) — so a run can be re-attached from the
    cluster after a GUI restart / network drop. The summary job is submitted
    *after* this manifest is staged, so it is not listed here.
    """
    job_ids = []
    templates = []
    for line in text.splitlines():
        parts = line.rstrip("\n").split(_SEP)
        if len(parts) < 4:
            continue
        kind, jobid, _label, directory = parts[:4]
        jobid = jobid.strip()
        if not jobid:
            continue
        if jobid not in job_ids:
            job_ids.append(jobid)
        if kind == "template":
            templates.append((directory, jobid))
    return job_ids, templates


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
        # Tab-separated fields (robust to a '|' in a job name); the format is
        # single-quoted so the remote shell keeps the literal tabs intact.
        fmt = _SEP.join(["%i", "%j", "%T", "%M", "%P", "%R"])
        out = self._run(
            "squeue -h -u "
            + shlex.quote(self.username)
            + " -o "
            + shlex.quote(fmt)
        )
        return parse_squeue(out)

    def recent_jobs(self, days=3):
        """The user's recent finished/running jobs (sacct)."""
        out = self._run(
            "sacct -X -n -P --delimiter="
            + shlex.quote(_SEP)
            + " -u "
            + shlex.quote(self.username)
            + f" -S now-{int(days)}days "
            + "-o JobID,JobName,State,Elapsed,End"
        )
        return parse_sacct(out)

    def jobs_status(self, job_ids):
        """``sacct`` status of specific ``job_ids`` (the current test run).

        One targeted ``sacct -j <ids>`` call — unlike :meth:`recent_jobs` it
        needs no look-back window and returns the given jobs in any phase
        (PENDING/RUNNING as well as finished), so the Test-runs tab can show a
        run's jobs the moment they are submitted. Blanks / the ``-`` skipped-
        tier placeholder are dropped; an empty id list is a no-op (``[]``,
        no SSH).
        """
        ids = [str(j) for j in job_ids if j and str(j) != "-"]
        if not ids:
            return []
        out = self._run(
            "sacct -X -n -P --delimiter="
            + shlex.quote(_SEP)
            + " -j "
            + shlex.quote(",".join(ids))
            + " -o JobID,JobName,State,Elapsed,End"
        )
        return parse_sacct(out)

    def list_test_runs(self, limit=30):
        """Recent test-run directories (newest first), for 'Attach to run'.

        Lists the ``<timestamp>_<version>`` subdirs of the configured
        ``apptainer_test_results_dir``. Returns full cluster paths (newest
        first, capped at ``limit``); ``[]`` when the dir is unset or empty.
        """
        results_dir = apptainer_test_results_dir(self.host)
        if not results_dir:
            return []
        # The trailing /*/ glob (outside the quotes) expands on the node to the
        # run subdirs; -1dt lists one per line, newest first. A no-match glob
        # errors, so swallow it -> an empty list rather than a raised error.
        out = self._run(
            "ls -1dt " + shlex.quote(results_dir) + "/*/ 2>/dev/null || true"
        )
        runs = [p.rstrip("/") for p in out.splitlines() if p.strip()]
        return runs[:limit]

    def attach_run(self, run_dir):
        """Re-attach to a previously submitted run from its ``jobs.tsv``.

        Reads ``<run_dir>/jobs.tsv`` and returns
        ``{run_dir, job_ids, templates}`` so the GUI can repopulate its tracked
        run after a restart / network drop — the run's identity lives on the
        cluster, not only in the GUI's memory.

        Raises
        ------
        RuntimeError
            If no ``jobs.tsv`` is found under ``run_dir`` (not a run directory,
            or a submit that never staged the manifest).
        """
        result = self.comm.execute_ssh_command(
            f"cat {shlex.quote(run_dir)}/jobs.tsv 2>/dev/null"
        )
        job_ids, templates = parse_jobs_tsv(result.get("stdout", "") or "")
        if not job_ids:
            raise RuntimeError(
                f"No jobs.tsv under {run_dir!r} — not a test-run directory "
                "(or the submit never staged it)."
            )
        return {
            "run_dir": run_dir,
            "job_ids": job_ids,
            "templates": templates,
        }

    def disk_usage(self):
        """``df -hP`` for the image directory (or ``None`` if unset).

        ``-P`` forces POSIX single-line output so a long network device name
        does not wrap onto its own line and defeat :func:`parse_df`.
        """
        image_dir, _module, _binds = apptainer_config(self.host)
        if not image_dir:
            return None
        out = self._run(f"df -hP {shlex.quote(image_dir)}")
        return parse_df(out)

    # -- test runs --------------------------------------------------------
    @staticmethod
    def _summarize_source():
        """The staged (stdlib-only) summarizer's text."""
        import importlib.resources

        return (
            importlib.resources.files("picasso_workflow.cluster_tests")
            .joinpath("summarize.py")
            .read_text()
        )

    def submit_test_chain(
        self,
        image_path,
        version,
        timestamp,
        test_data_dir=None,
        run_tiers13=True,
        run_tier4=True,
        run_templates=True,
        template_time=None,
    ):
        """Stage + submit the container test chain; return a summary dict.

        ``image_path`` is the (test) ``.sif`` to run pytest inside; ``version``
        labels the run dir; ``timestamp`` tags it (passed in so the caller
        controls the clock). ``template_time`` (a SLURM walltime like
        ``"04:00:00"``) caps each template workflow run via ``sbatch --time``
        so a hung one is killed and the summary job unblocks; ``None``/blank
        leaves each template script's own ``--time`` untouched. See
        :class:`testrun.TestChainSubmitter`.
        """
        from picasso_workflow.admin.testrun import TestChainSubmitter
        from picasso_workflow.util import convert_filepath_for_machine

        # The test-data dir is configured / typed in the GUI machine's layout
        # (e.g. a Mac "/Volumes/..." mount), but `find` + the apptainer binds
        # run on the cluster login node. Translate it to the host's layout via
        # the Drivepaths config (dest_machine = the cluster key) so it resolves
        # there. A path under no known drive root, or an unmapped host, is
        # returned unchanged.
        if test_data_dir:
            # Drivepaths rewriting is a best-effort convenience; guard it so any
            # unexpected failure in the shared PathParser degrades to "use the
            # entered path" (which then resolves as-is or fails later with a
            # clear `find` error) rather than aborting the submit with an opaque
            # traceback at this GUI boundary.
            try:
                test_data_dir = convert_filepath_for_machine(
                    test_data_dir, dest_machine=self.host
                )
            except Exception as exc:
                logger.warning(
                    "Drivepaths conversion of {!r} for host {!r} failed "
                    "({}); using the entered path unchanged.",
                    test_data_dir,
                    self.host,
                    exc,
                )

        image_dir, module, bind_paths = apptainer_config(self.host)
        submitter = TestChainSubmitter(
            self,
            image_path=image_path,
            test_results_dir=apptainer_test_results_dir(self.host),
            module=module,
            bind_paths=bind_paths,
            summarize_source=self._summarize_source(),
        )
        return submitter.submit(
            version=version,
            timestamp=timestamp,
            test_data_dir=test_data_dir,
            run_tiers13=run_tiers13,
            run_tier4=run_tier4,
            run_templates=run_templates,
            template_time=template_time,
        )

    def fetch_summary(self, run_dir):
        """Return ``<run_dir>/SUMMARY.txt`` if present, else ``""``."""
        result = self.comm.execute_ssh_command(
            f"cat {shlex.quote(run_dir)}/SUMMARY.txt 2>/dev/null"
        )
        return result.get("stdout", "") or ""

    def fetch_logs(self, run_dir, templates=None, tail=400):
        """Return the per-job ``.log`` files for a run (each tailed).

        The tier/summary jobs write ``<label>_<jobid>.log`` into ``run_dir``;
        ``templates`` (a list of ``(folder, job_id)`` from a submit's
        ``template_jobs``) additionally pulls each template's own
        ``<folder>/logs/*<job_id>*`` -- so a failed end-to-end template run's
        error output is visible from the console without hand-SSH. Each matched
        file is tailed to the last ``tail`` lines under a ``===== path =====``
        header. Returns ``""`` when nothing matches yet (a no-match glob is
        skipped, not catted literally).
        """
        # Each glob's `*` stays outside the quotes so it still expands; the
        # fixed parts (run_dir / template folder) are shell-quoted.
        globs = [f"{shlex.quote(run_dir)}/*.log"]
        for folder, job_id in templates or []:
            globs.append(
                f"{shlex.quote(folder)}/logs/*{shlex.quote(str(job_id))}*"
            )
        # `for f in <globs>` iterates the literal pattern when nothing matches,
        # so guard each entry with `[ -e ]` before tailing.
        cmd = (
            f"for f in {' '.join(globs)}; do "
            '[ -e "$f" ] || continue; '
            'printf "\\n===== %s =====\\n" "$f"; '
            f'tail -n {int(tail)} "$f" 2>/dev/null || true; '
            "done"
        )
        # Use ``_run`` (not a raw ``execute_ssh_command``): the remote command
        # always exits 0 (no matching files -> loop body never runs; a single
        # unreadable log is swallowed by ``|| true``), so a non-zero exit is a
        # genuine SSH / auth failure (ssh itself returns 255) that must surface
        # as an error -- not be masked as the empty "(no logs yet)" string the
        # caller shows on "".
        return self._run(cmd)

    def cancel_jobs(self, job_ids):
        """``scancel`` the given job ids; return the number cancelled.

        Filters out blanks / the ``-`` placeholder (so a skipped tier's
        ``None`` job never reaches ``scancel``) and is a no-op returning ``0``
        when nothing valid remains. One ``scancel`` call cancels them all --
        ``scancel`` exits 0 even for already-finished ids, so a stuck run can be
        cleared after a gating tier fails.
        """
        ids = [str(j) for j in job_ids if j and str(j) != "-"]
        if not ids:
            return 0
        self._run("scancel " + " ".join(shlex.quote(i) for i in ids))
        return len(ids)
