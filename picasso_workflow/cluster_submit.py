#!/usr/bin/env python
"""WP-9 Phase C -- the cluster-submit worker (sbatch-over-SSH, undrift-first).

A non-GUI worker that hands one QC-passed dataset to the SLURM cluster. It is
the consumer end of the WP-9 hand-off state machine (``ready -> claimed ->
running -> done/failed``) the registry owns: QC readies a dataset
(``mark_ready``), this worker atomically CLAIMS it (``claim_handoff`` -- exactly
once; a second worker claiming the same ``ready`` dataset gets ``None``), stages
its localizations to cluster scratch, records the cluster aggregation as an
``analysis_run`` in the registry (carrying the WP-TEMPLATE **Phase 3**
conformance of the submitted workflow), submits an sbatch job that
undrift-corrects then aggregates inside the dev-container image via Apptainer,
and transitions the hand-off ``claimed -> running``. The cluster job itself
closes the loop (``running -> done``/``failed``) via :func:`cluster_writeback`.

What crosses to the cluster (C33/C15): only the **localizations** -- never the
raw movie. The authoritative localization pass already ran on the acquisition
side with no drift correction, so the cluster recipe STARTS WITH undrift
(:func:`picasso_workflow.standard_aggregation_workflows.cluster_aggregation`).

Reuse, not reimplementation
---------------------------
The SSH + sbatch-over-SSH transport is the GUI's existing ``SlurmCommunicator``,
now in the Qt-free :mod:`picasso_workflow.slurm_submit` module (factored out of
``gui.py`` so this worker can reuse it without importing PyQt). Local->remote
path remapping is ``metaworkflow.PathParser.convert_path``. Both are injectable
so tests mock the SSH/scp runner and never touch the network.

Registry coupling is LAZY (picasso-registry is not a picasso-workflow
dependency -- mirrors ``backfill/ingest.py``): the worker takes a registry
*client* object (real ``RegistryClient`` or the in-memory test double); nothing
here imports picasso-registry at module load, so the package still imports
headlessly on the wheel-only CI.
"""

from __future__ import annotations

import datetime as _dt
import shlex
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import yaml
from loguru import logger

import picasso_workflow.standard_aggregation_workflows as saw
from picasso_workflow import templates as _templates

# Hand-off states (mirror the registry state machine; local constants so the
# worker never hard-codes a bare string in flow control).
STATE_READY = "ready"
STATE_CLAIMED = "claimed"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"

#: Tag stamped into the analysis_run so cluster-submit-created rows are
#: distinguishable from live/preprocess/backfill rows on the same dataset.
CLUSTER_SUBMIT_TAG = "wp9-cluster-submit"
ANALYSIS_KIND = "cluster-aggregation"
COMPUTE_LOCATION = "cluster"

#: Apptainer module pinned to the LOWEST version on the target node group
#: (register C10: hpcl8 max 1.4.1; hpcl10/hpcl20 up to 1.5.2 -- WP-9 pins to the
#: lowest so the same submit template runs on every node group). Overridable per
#: deployment via ``ClusterSubmitConfig.apptainer_module``.
DEFAULT_APPTAINER_MODULE = "apptainer/1.4.1"


# ---------------------------------------------------------------------------
# Workflow: the undrift-first cluster aggregation.
# ---------------------------------------------------------------------------
def build_cluster_workflow(
    filepaths: list[str], *, undrift: str = "aim", segmentation: int = 500
) -> list[tuple[str, dict]]:
    """Return the resolved undrift-first cluster aggregation workflow.

    A flat ``[(module, params), ...]`` list: the per-channel single-dataset
    leg (``load_dataset_localizations -> undrift_{aim,rcc} -> save``) followed
    by the aggregation leg. Its first real processing module is the undrift, so
    "starts with undrift" is explicit in the submitted workflow and in its
    template-conformance diff.
    """
    built = build_cluster_aggregation_spec(
        filepaths, undrift=undrift, segmentation=segmentation
    )
    return list(built.get("single_dataset_modules", [])) + list(
        built.get("aggregation_modules", [])
    )


def build_cluster_aggregation_spec(
    filepaths: list[str], *, undrift: str = "aim", segmentation: int = 500
) -> dict:
    """Return the full aggregation spec dict the cluster runner consumes.

    The ``{single_dataset_tileparameters, single_dataset_modules,
    aggregation_modules}`` mapping ``AggregationWorkflowRunner.config_from_dicts``
    takes (see :func:`cluster_run`). ``build_cluster_workflow`` flattens this to
    an ordered ``[(module, params)]`` list for the template-conformance diff;
    the full dict is what gets staged + executed.
    """
    return saw.cluster_aggregation(
        filepaths, undrift=undrift, segmentation=segmentation
    )


def _yaml_safe(value: Any) -> Any:
    """Recursively turn tuples into lists so ``yaml.safe_dump`` can serialize.

    Module params carry recipe command tuples (e.g. ``("$$map", "filepath")``)
    that the safe dumper has no representer for. Mirrors the GUI's
    ``Window._yaml_safe`` so the staged spec round-trips. (On load the runner
    treats list-form directives the same as tuple-form.)
    """
    if isinstance(value, tuple):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, list):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    return value


def _conformance_for(workflow: list[tuple[str, dict]]) -> dict:
    """WP-TEMPLATE Phase-3 conformance of ``workflow`` vs the cluster template.

    Diffs the submitted workflow against the registered ``cluster_aggregation``
    template. An exactly-default submission is ``exact``; a param override or
    an added/removed module is reported accordingly -- the same record WP-6
    POSTs, here attached into the analysis_run at creation (see
    :meth:`ClusterSubmitWorker._create_analysis_run`).
    """
    template = _templates.get_template("cluster_aggregation")
    return _templates.template_conformance_record(workflow, template)


# ---------------------------------------------------------------------------
# sbatch script: apptainer module-load pin + undrift-first run.
# ---------------------------------------------------------------------------
def build_sbatch_script(
    *,
    job_name: str,
    workflow_yaml_remote: str,
    results_dir_remote: str,
    image_path: str,
    apptainer_module: str = DEFAULT_APPTAINER_MODULE,
    bind_paths: Optional[list[str]] = None,
    partition: Optional[str] = None,
    nodelist: Optional[str] = None,
    constraint: Optional[str] = None,
    time_limit: str = "08:00:00",
    mem: str = "64G",
    cpus_per_task: int = 8,
) -> str:
    """Build the sbatch script string: module-load Apptainer, undrift-first run.

    The analysis stack runs INSIDE the dev-container image via Apptainer: the
    script ``module load``s the pinned Apptainer version FIRST (``apptainer/
    1.4.1`` by default -- the lowest on the target node group, C10), then
    ``apptainer exec``s the image to run the recommended undrift-first
    aggregation over the staged ``workflow.yaml``. The node group is constrained
    via ``--partition`` / ``--nodelist`` / ``-C`` (whichever the deployment
    sets). ``PW_RC`` carries the run's exit status out as the job's exit code,
    so a failed run reports FAILED to SLURM (not a bare 0).
    """
    binds = bind_paths or [results_dir_remote]
    # Every remote path lands in a shell command (the apptainer invocation) or a
    # SLURM directive value; lab data folders routinely contain spaces (e.g.
    # "channel 0.hdf5"), so each is shlex-quoted or an unquoted space would
    # word-split the argv (--spec <half-a-path>) / break the #SBATCH directive.
    # Mirrors SlurmCommunicator.fetch_progress et al., which already quote.
    bind_arg = " ".join(f"--bind {shlex.quote(b)}" for b in binds)
    out_pattern = shlex.quote(f"{results_dir_remote}/{job_name}_%j.out")
    err_pattern = shlex.quote(f"{results_dir_remote}/{job_name}_%j.err")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={shlex.quote(job_name)}",
        f"#SBATCH --output={out_pattern}",
        f"#SBATCH --error={err_pattern}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={shlex.quote(partition)}")
    if nodelist:
        lines.append(f"#SBATCH --nodelist={shlex.quote(nodelist)}")
    if constraint:
        lines.append(f"#SBATCH --constraint={shlex.quote(constraint)}")
    run_cmd = (
        "apptainer exec "
        + bind_arg
        + f" {shlex.quote(image_path)} "
        + "python -m picasso_workflow.cluster_submit run "
        + f"--spec {shlex.quote(workflow_yaml_remote)} "
        + f"--results-dir {shlex.quote(results_dir_remote)}"
    )
    lines += [
        "",
        'echo "Job started at: $(date)"',
        'echo "Running on node: $(hostname)"',
        "",
        "source /etc/profile.d/modules.sh",
        # Apptainer version PINNED to the lowest on the target node group
        # (C10). module-load FIRST so `apptainer` resolves below.
        f"module load {shlex.quote(apptainer_module)}",
        "",
        "# Run the recommended undrift-first aggregation inside the "
        "dev-container image. The 'run' entry point loads the staged "
        "aggregation spec and executes AggregationWorkflowRunner.",
        run_cmd,
        "PW_RC=$?",
        "",
        'echo "Job completed at: $(date) (exit ${PW_RC:-0})"',
        "exit ${PW_RC:-0}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
@dataclass
class ClusterSubmitConfig:
    """Deployment config for the cluster-submit worker.

    Everything the worker needs to stage + submit, kept as data so a real run
    is a config swap (real SSH host/creds, real scratch paths), not a code
    change. Tests pass a throwaway config and inject mocks for the SSH
    communicator + stager.
    """

    #: SLURM login node hostname (sbatch-over-SSH target).
    host: str = "hpcl8"
    #: Destination machine key for PathParser local->remote remapping.
    dest_machine: str = "hpcl8"
    #: Cluster scratch root the localizations are staged into.
    scratch_root: str = "/scratch/paint"
    #: Path to the dev-container Apptainer image on the cluster.
    image_path: str = "/sw/picasso-workflow/picasso-workflow.sif"
    apptainer_module: str = DEFAULT_APPTAINER_MODULE
    partition: Optional[str] = None
    nodelist: Optional[str] = None
    constraint: Optional[str] = None
    bind_paths: Optional[list[str]] = None
    undrift: str = "aim"
    segmentation: int = 500
    #: Extra per-channel bind paths / rsync options the stager may read.
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The worker.
# ---------------------------------------------------------------------------
class ClusterSubmitWorker:
    """Claim one ready dataset and hand it to the SLURM cluster.

    Parameters
    ----------
    registry : object
        A registry client exposing the WP-9 hand-off surface (``claim_handoff``
        / ``transition_handoff`` / ``log_analysis``) -- the real
        ``RegistryClient`` or the in-memory test double. Imported LAZILY by the
        caller (picasso-registry is not a picasso-workflow dep), so this module
        never imports it.
    config : ClusterSubmitConfig
        Deployment paths/host/apptainer settings.
    communicator : object, optional
        A ``SlurmCommunicator``-shaped object (``submit_job``). Default builds
        the real :class:`picasso_workflow.slurm_submit.SlurmCommunicator` for
        ``config.host``; tests inject a mock SSH runner.
    stager : callable, optional
        ``stager(local_paths, remote_dir, config) -> list[str]`` copies the
        localizations to cluster scratch and returns their remote paths.
        Default :func:`rsync_stage` (shells out over the communicator); tests
        inject a no-network double.
    worker_id : str, optional
        The claim owner recorded in the hand-off (default: host/pid).
    """

    def __init__(
        self,
        registry: Any,
        config: Optional[ClusterSubmitConfig] = None,
        *,
        communicator: Any | None = None,
        stager: Optional[Callable[..., list[str]]] = None,
        worker_id: Optional[str] = None,
    ) -> None:
        self.registry = registry
        self.config = config or ClusterSubmitConfig()
        self._communicator = communicator
        self.stager = stager or rsync_stage
        self.worker_id = worker_id or _default_worker_id()

    # -- transport (lazy; Qt-free) -----------------------------------------
    @property
    def communicator(self) -> Any:
        """The SSH/sbatch transport, built once for ``config.host``."""
        if self._communicator is None:
            from picasso_workflow.slurm_submit import SlurmCommunicator

            self._communicator = SlurmCommunicator(
                self.config.host, _default_worker_id().split("@")[0]
            )
        return self._communicator

    def _remote_path(self, local_path: str) -> str:
        """Local->remote remap via PathParser (shared with the GUI submit)."""
        from picasso_workflow.metaworkflow import PathParser

        return PathParser().convert_path(local_path, self.config.dest_machine)

    # -- the flow ----------------------------------------------------------
    def run_once(
        self, *, acquisition_run_id: Optional[str] = None
    ) -> Optional[dict]:
        """Claim one ready dataset and submit it. Returns a summary, or None.

        ``None`` means there was nothing to do (no ready dataset / lost the
        claim race) -- a clean no-op. On a win, runs the full
        stage -> analysis_run (+ Phase-3 conformance) -> submit -> transition
        flow and returns a summary dict. Any failure AFTER a successful claim
        marks the hand-off FAILED (with a reason), so a claimed dataset is
        never silently lost.
        """
        claimed = self.registry.claim_handoff(
            self.worker_id, acquisition_run_id=acquisition_run_id
        )
        if not claimed:
            logger.info("cluster-submit: nothing ready to claim.")
            return None

        acq_id = claimed["acquisition_run_id"]
        logger.info(f"cluster-submit: claimed {acq_id} as {self.worker_id}.")
        try:
            return self._submit_claimed(claimed)
        except Exception as exc:  # never lose a claimed dataset
            logger.error(
                f"cluster-submit: submit failed for {acq_id}: {exc!r}; "
                "marking hand-off FAILED for re-ready."
            )
            self._fail(
                acq_id, reason=repr(exc), base_detail=claimed.get("detail")
            )
            return {
                "acquisition_run_id": acq_id,
                "submitted": False,
                "error": repr(exc),
            }

    def _submit_claimed(self, claimed: dict) -> dict:
        acq_id = claimed["acquisition_run_id"]
        local_locs = self._dataset_locs(claimed)

        # 1) Stage localizations to cluster scratch (locs only -- never movies).
        remote_dir = f"{self.config.scratch_root}/{acq_id}"
        remote_locs = self.stager(local_locs, remote_dir, self.config)

        # 2) Build the recommended undrift-first aggregation over the staged
        #    locs: the full spec (staged + executed) and its flattened form
        #    (for the Phase-3 conformance diff + the analysis_run record).
        spec = build_cluster_aggregation_spec(
            remote_locs,
            undrift=self.config.undrift,
            segmentation=self.config.segmentation,
        )
        workflow = list(spec.get("single_dataset_modules", [])) + list(
            spec.get("aggregation_modules", [])
        )
        conformance = _conformance_for(workflow)

        # 3) Create the cluster analysis_run (carrying the conformance in
        #    extra), then move the hand-off claimed -> running with its id.
        analysis_run_id = self._create_analysis_run(
            acq_id, workflow, conformance
        )
        moved = self.registry.transition_handoff(
            acq_id,
            expected=STATE_CLAIMED,
            to=STATE_RUNNING,
            worker=self.worker_id,
            analysis_run_id=analysis_run_id,
        )
        if moved is None:
            # Lost the claimed->running guard (someone else moved it). Do not
            # submit; leave the registry authoritative.
            raise RuntimeError(
                f"claimed->running transition rejected for {acq_id}"
            )

        # 4) Write the spec yaml + sbatch script, submit over SSH.
        job = self._submit_job(acq_id, spec, remote_dir, analysis_run_id)
        if not job.get("success"):
            raise RuntimeError(
                f"sbatch submission failed: {job.get('stderr')!r}"
            )

        logger.info(
            f"cluster-submit: {acq_id} submitted "
            f"(job {job.get('job_id')}, analysis_run {analysis_run_id})."
        )
        return {
            "acquisition_run_id": acq_id,
            "analysis_run_id": analysis_run_id,
            "job_id": job.get("job_id"),
            "conformance": conformance["conformance"],
            "submitted": True,
        }

    # -- steps -------------------------------------------------------------
    def _dataset_locs(self, claimed: dict) -> list[str]:
        """Local localization file paths for the claimed dataset.

        Prefers an explicit ``detail.locs`` list carried on the hand-off
        (whatever QC readied it with); falls back to a single
        ``detail.locs_path``. If the hand-off detail carries no locs pointer
        (e.g. a re-readied dataset — ``reready_handoff`` resets ``detail`` to
        None, so the pointer does NOT survive a failure+requeue), it falls
        back to the acquisition_run's durable, append-only ``extra.locs`` /
        ``extra.locs_path``. Without a reachable pointer anywhere the dataset
        cannot be staged, which is a hard error (never a silent skip).
        """
        detail = claimed.get("detail") or {}
        locs = detail.get("locs")
        if locs:
            return list(locs)
        one = detail.get("locs_path")
        if one:
            return [one]

        # Durable fallback: the acquisition_run's append-only extra survives a
        # reready that wiped the mutable hand-off detail. Best-effort lookup;
        # a registry/lookup failure falls through to the hard error below.
        acq_id = claimed.get("acquisition_run_id")
        if acq_id:
            try:
                acq = self.registry.get("acquisition_run", acq_id)
            except Exception as exc:
                logger.warning(
                    f"cluster-submit: could not read acquisition_run "
                    f"{acq_id!r} for locs fallback: {exc!r}"
                )
                acq = None
            extra = (acq or {}).get("extra") or {}
            locs = extra.get("locs")
            if locs:
                return list(locs)
            one = extra.get("locs_path")
            if one:
                return [one]

        raise ValueError(
            f"no localization pointer for hand-off {claimed.get('id')!r} "
            "(detail.locs / detail.locs_path / acquisition_run.extra.locs)"
        )

    def _create_analysis_run(
        self,
        acquisition_run_id: str,
        workflow: list[tuple[str, dict]],
        conformance: dict,
    ) -> str:
        """Create the cluster aggregation's analysis_run (+ Phase-3 sink).

        WP-TEMPLATE **Phase 3**: the conformance record is attached into the
        analysis_run at creation under ``extra["template_conformance"]``. The
        store is append-only / set-at-creation, so this is the one place to
        wire it -- no migration, additive. Returns the (server-minted)
        analysis_run id.
        """
        workflow_yaml = yaml.safe_dump(
            [{"module": m, "params": p} for m, p in workflow],
            sort_keys=False,
        )
        row = self.registry.log_analysis(
            acquisition_run_id=acquisition_run_id,
            kind=ANALYSIS_KIND,
            compute_location=COMPUTE_LOCATION,
            status=STATE_RUNNING,
            recommender_version=CLUSTER_SUBMIT_TAG,
            workflow_yaml=workflow_yaml,
            parameters={"workflow": [m for m, _ in workflow]},
            extra={"template_conformance": conformance},
            started_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        )
        return row["id"]

    def _submit_job(
        self,
        acquisition_run_id: str,
        spec: dict,
        remote_dir: str,
        analysis_run_id: str,
    ) -> dict:
        """Write the aggregation spec yaml + sbatch script, sbatch it.

        The staged ``workflow.yaml`` holds the full aggregation spec plus the
        reporter/analysis config the cluster runner (:func:`cluster_run`) loads;
        ``analysis_config.registry.run_id`` carries the analysis_run id so the
        cluster-side per-module logging and the write-back join the same run.
        """
        comm = self.communicator
        job_name = f"pwf-agg-{acquisition_run_id}"

        # The cluster runner (AggregationWorkflowRunner.config_from_dicts)
        # REQUIRES reporter_config["report_name"] and
        # analysis_config["result_location"] (both indexed unconditionally), so
        # empty dicts would KeyError on the first line of the run. Supply a
        # minimal, Confluence-less config: results land under the run's scratch
        # dir, the report is named by the dataset. The registry run_id rides in
        # analysis_config so cluster-side logging + the write-back join this run.
        staged = {
            "acquisition_run_id": acquisition_run_id,
            "analysis_run_id": analysis_run_id,
            "aggregation_workflow": _yaml_safe(spec),
            "reporter_config": {
                "report_name": f"cluster-agg-{acquisition_run_id}",
            },
            "analysis_config": {
                "result_location": remote_dir,
                "registry": {"run_id": analysis_run_id},
            },
        }
        workflow_yaml = yaml.safe_dump(staged, sort_keys=False)
        workflow_remote = f"{remote_dir}/workflow.yaml"
        comm.execute_ssh_command(f"mkdir -p {shlex.quote(remote_dir)}")
        _ssh_write_file(comm, workflow_remote, workflow_yaml)

        script = build_sbatch_script(
            job_name=job_name,
            workflow_yaml_remote=workflow_remote,
            results_dir_remote=remote_dir,
            image_path=self.config.image_path,
            apptainer_module=self.config.apptainer_module,
            bind_paths=self.config.bind_paths,
            partition=self.config.partition,
            nodelist=self.config.nodelist,
            constraint=self.config.constraint,
        )
        script_remote = f"{remote_dir}/submit.sh"
        _ssh_write_file(comm, script_remote, script)

        return comm.submit_job(script_remote, self.config.dest_machine)

    def _fail(
        self,
        acquisition_run_id: str,
        *,
        reason: str,
        base_detail: Optional[dict] = None,
    ) -> None:
        """Best-effort claimed/running -> failed (never raises).

        A claimed dataset whose submission broke must land in ``failed`` (so
        ``reready_handoff`` can re-queue it) rather than be lost. We try both
        the ``claimed`` and ``running`` source states, since the break may have
        happened on either side of the transition. The failure reason is
        MERGED into the hand-off's existing ``detail`` (the locs pointers QC
        readied it with) rather than overwriting it -- ``transition_handoff``
        replaces ``detail`` wholesale, so dropping the locs here would strand
        the dataset on re-ready.
        """
        detail = dict(base_detail or {})
        detail.update({"error": reason, "stage": "cluster-submit"})
        for expected in (STATE_RUNNING, STATE_CLAIMED):
            try:
                row = self.registry.transition_handoff(
                    acquisition_run_id,
                    expected=expected,
                    to=STATE_FAILED,
                    worker=self.worker_id,
                    detail=detail,
                )
                if row is not None:
                    return
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(
                    f"cluster-submit: fail-transition from {expected} "
                    f"errored for {acquisition_run_id}: {exc!r}"
                )


# ---------------------------------------------------------------------------
# Staging.
# ---------------------------------------------------------------------------
def rsync_stage(
    local_paths: list[str],
    remote_dir: str,
    config: ClusterSubmitConfig,
) -> list[str]:  # pragma: no cover - real network path, mocked in tests
    """rsync the localizations to ``remote_dir`` on the cluster; return remotes.

    Only the localization files cross (C33/C15) -- the caller passes locs
    paths, never movies. Shells out to ``rsync`` over SSH; emulator-validated
    here (tests inject a no-network stager). A real run swaps in real SSH creds
    via ``config``, not code.
    """
    import subprocess

    remote_paths = []
    # Quote the remote path for the REMOTE shell: `ssh host "<cmd>"` runs <cmd>
    # through the login shell, and lab folders contain spaces.
    subprocess.run(
        ["ssh", config.host, f"mkdir -p {shlex.quote(remote_dir)}"],
        check=True,
    )
    for p in local_paths:
        base = p.rsplit("/", 1)[-1]
        dest = f"{remote_dir}/{base}"
        # `-s` (--protect-args) stops rsync from re-splitting the remote path on
        # spaces (otherwise "channel 0.hdf5" becomes two args on the far side).
        subprocess.run(
            ["rsync", "-a", "-s", p, f"{config.host}:{dest}"], check=True
        )
        remote_paths.append(dest)
    return remote_paths


# ---------------------------------------------------------------------------
# Cluster-side write-back (the other half of the contract).
# ---------------------------------------------------------------------------
def cluster_writeback(
    registry: Any,
    acquisition_run_id: str,
    *,
    success: bool,
    metrics: Optional[dict] = None,
    analysis_run_id: Optional[str] = None,
    detail: Optional[dict] = None,
) -> Optional[dict]:
    """Close the loop from the cluster job: ``running -> done``/``failed``.

    The write-back contract (runs on the cluster, after the aggregation): on
    success, transition the hand-off ``running -> done`` and post final
    metrics; on failure, ``running -> failed`` with a ``detail`` reason. The
    guard means an illegal transition (not currently ``running``) returns
    ``None`` (the registry 409s) rather than corrupting state. Metrics posting
    is best-effort -- a metrics hiccup never blocks the state transition.

    Emulated in tests (the real cluster run is a user-side smoke test): a test
    calls this against the in-memory registry to prove the ``done``/``failed``
    transitions and the illegal-transition 409.
    """
    to = STATE_DONE if success else STATE_FAILED
    row = registry.transition_handoff(
        acquisition_run_id,
        expected=STATE_RUNNING,
        to=to,
        analysis_run_id=analysis_run_id,
        detail=detail,
    )
    if row is None:
        logger.warning(
            f"cluster-writeback: {acquisition_run_id} not in 'running'; "
            f"transition to {to!r} rejected (409)."
        )
        return None
    if success and metrics and analysis_run_id:
        try:
            registry.log_metrics(
                analysis_run_id=analysis_run_id,
                scope="aggregation",
                **metrics,
            )
        except Exception as exc:  # metrics never block the state machine
            logger.warning(
                f"cluster-writeback: metrics post failed for "
                f"{acquisition_run_id}: {exc!r}"
            )
    return row


# ---------------------------------------------------------------------------
# helpers.
# ---------------------------------------------------------------------------
def _default_worker_id() -> str:
    import os
    import socket

    return f"{socket.gethostname()}@{os.getpid()}"


def _ssh_write_file(comm: Any, remote_path: str, content: str) -> None:
    """Write ``content`` to ``remote_path`` via the communicator's SSH runner.

    Uses a heredoc over ``execute_ssh_command`` so no local temp file / scp
    round-trip is needed; the communicator is the single SSH chokepoint tests
    mock. The target path is shlex-quoted (lab paths contain spaces) and the
    heredoc marker is quoted so the content is written literally (no shell
    expansion of ``$``/backtick in the yaml or script body).
    """
    cmd = f"cat > {shlex.quote(remote_path)} <<'PWF_EOF'\n{content}\nPWF_EOF"
    comm.execute_ssh_command(cmd)


# ---------------------------------------------------------------------------
# Cluster-side entry point (runs INSIDE the apptainer image on the node).
# ---------------------------------------------------------------------------
#: Env vars the cluster-side runner reads to build the real registry client
#: for the write-back (same names as backfill, so one deployment configures all).
SERVICE_URL_ENV = "PAINT_REGISTRY_SERVICE_URL"
TOKEN_ENV = "PAINT_REGISTRY_TOKEN"


def _registry_from_env() -> Optional[Any]:
    """Build the real RegistryClient from env, or ``None`` if not configured.

    Lazy (picasso-registry is not a dep): on the cluster the write-back uses the
    real client; off a configured service (``PAINT_REGISTRY_SERVICE_URL`` unset)
    it returns ``None`` and the write-back is skipped -- the run still executes.
    """
    import os

    url = os.environ.get(SERVICE_URL_ENV)
    if not url:
        return None
    try:
        from picasso_registry.client import RegistryClient

        return RegistryClient(base_url=url, token=os.environ.get(TOKEN_ENV))
    except Exception as exc:  # never block the run on a client-build failure
        logger.warning(
            f"cluster-run: could not build registry client: {exc!r}"
        )
        return None


def cluster_run(
    spec_path: str,
    *,
    results_dir: Optional[str] = None,
    registry: Any | None = None,
) -> bool:
    """Run the staged undrift-first aggregation ON THE CLUSTER, then write back.

    The sbatch script's entry point (``python -m picasso_workflow.cluster_submit
    run --spec <workflow.yaml>``). Loads the staged spec, runs the
    ``AggregationWorkflowRunner``, and on completion transitions the hand-off
    ``running -> done`` (or ``-> failed`` with a reason) via
    :func:`cluster_writeback` -- the other half of the write-back contract.

    ROUND-TRIP: executed on the cluster only (a user-side smoke test). Here it
    is import-validated + unit-tested at the boundary (:func:`cluster_writeback`
    is tested directly against the in-memory registry); the full
    runner.run() path is a real-cluster smoke test.
    """
    import os

    with open(spec_path, encoding="utf-8") as fh:
        staged = yaml.safe_load(fh)

    acquisition_run_id = staged.get("acquisition_run_id")
    analysis_run_id = staged.get("analysis_run_id")
    reg = registry if registry is not None else _registry_from_env()

    # Import the runner lazily: it pulls the heavy analysis stack, which the
    # submit-side worker (and its hermetic tests) must not import.
    from picasso_workflow import AggregationWorkflowRunner

    analysis_config = dict(staged.get("analysis_config") or {})
    # ``result_location`` is the key the runner indexes (not ``result_folder``);
    # default it to the sbatch --results-dir if the staged config omitted it.
    if results_dir:
        analysis_config.setdefault("result_location", results_dir)
    result_location = analysis_config.get("result_location")
    if result_location:
        os.makedirs(result_location, exist_ok=True)

    ok = False
    reason = None
    try:
        runner = AggregationWorkflowRunner.config_from_dicts(
            reporter_config=staged.get("reporter_config") or {},
            analysis_config=analysis_config,
            aggregation_workflow=staged["aggregation_workflow"],
        )
        result = runner.run()
        ok = result is not False
    except Exception as exc:  # the job failed; record why on the hand-off
        reason = repr(exc)
        logger.error(f"cluster-run: aggregation failed: {reason}")

    if reg is not None and acquisition_run_id:
        cluster_writeback(
            reg,
            acquisition_run_id,
            success=ok,
            analysis_run_id=analysis_run_id,
            detail=None if ok else {"reason": reason, "stage": "cluster-run"},
        )
    return ok


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: ``python -m picasso_workflow.cluster_submit run --spec <yaml>``."""
    import argparse

    parser = argparse.ArgumentParser(prog="picasso_workflow.cluster_submit")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run a staged cluster aggregation spec")
    run_p.add_argument("--spec", required=True, help="path to workflow.yaml")
    run_p.add_argument("--results-dir", default=None)
    args = parser.parse_args(argv)

    if args.command == "run":
        ok = cluster_run(args.spec, results_dir=args.results_dir)
        return 0 if ok else 1
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry
    import sys

    sys.exit(main())
