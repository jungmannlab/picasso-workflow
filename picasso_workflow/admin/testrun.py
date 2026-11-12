"""Container-based cluster test-chain orchestration for the admin console.

Ports ``tools/cluster_tests/submit_all.sh`` to Python driving the existing
:class:`picasso_workflow.slurm_submit.SlurmCommunicator`, so the tiered test
run (tiers 1-3 -> tier 4 + per-template workflows -> summary, with ``afterok``
/ ``afterany`` dependency chaining) can be launched from the GUI and runs
**inside an Apptainer image** -- no cloned repo or conda env on the login node.

The test code + fixtures come from the image (``pytest --pyargs
picasso_workflow``); the summarizer (stdlib-only) is staged into the run
directory and runs with the node's ``python3`` (where ``sacct`` is available --
it is not inside the container).

The command/script builders at the top are pure functions (no SSH), so the
risky assembly -- apptainer exec with the right binds, ``--env`` cache dirs, and
explicit ``-m`` markers -- is unit-tested; :class:`TestChainSubmitter` is the
thin SSH orchestration over them.
"""

from __future__ import annotations

import posixpath
import shlex

from picasso_workflow.slurm_submit import apptainer_exec_prefix

# Per-tier SLURM resources (mirroring the tools/cluster_tests/*.sbatch files).
_RES_TIERS = {
    "nodes": 1,
    "ntasks": 1,
    "cpus-per-task": 8,
    "mem": "8G",
    "time": "00:45:00",
}
_RES_TIER4 = {
    "nodes": 1,
    "ntasks": 1,
    "cpus-per-task": 24,
    "mem": "50G",
    "time": "12:00:00",
}
_RES_SUMMARY = {
    "nodes": 1,
    "ntasks": 1,
    "cpus-per-task": 1,
    "mem": "2G",
    "time": "00:05:00",
}


# ---------------------------------------------------------------------------
# Pure builders (no SSH)
# ---------------------------------------------------------------------------
def run_dir_path(test_results_dir, version, timestamp):
    """The per-run cluster directory ``<test_results_dir>/<ts>_<version>``."""
    tag = f"{timestamp}_{version or 'dev'}"
    return posixpath.join(test_results_dir, tag)


def cache_env(run_dir):
    """Writable cache/config dirs so picasso imports inside a read-only SIF.

    ``@numba.njit(cache=True)`` and matplotlib otherwise try to write next to
    their read-only in-image package files and fail; point them at the run
    directory (bound in) instead.
    """
    return {
        "NUMBA_CACHE_DIR": posixpath.join(run_dir, ".numba"),
        "MPLCONFIGDIR": posixpath.join(run_dir, ".mpl"),
    }


def pytest_command(
    image_path,
    run_dir,
    marker,
    junit_label,
    bind_paths,
    test_data_dir=None,
):
    """One ``apptainer exec ... pytest`` invocation for a tier.

    Runs the suite FROM THE IMAGE (``--pyargs picasso_workflow``) with an
    explicit ``-m`` marker (pyproject's default-deselect ``addopts`` is not
    shipped in the image). The JUnit path embeds ``${SLURM_JOB_ID}`` (left
    unquoted so the node shell expands it) and lands in the bound run dir, named
    so the summarizer finds it (``<label>_<jobid>.xml``). ``test_data_dir`` (for
    tier 4) is bound in and exported as ``PW_TEST_DATA_DIR``.
    """
    binds = list(bind_paths) + [run_dir]
    env = dict(cache_env(run_dir))
    if test_data_dir:
        binds.append(test_data_dir)
        env["PW_TEST_DATA_DIR"] = test_data_dir
    prefix = apptainer_exec_prefix(image_path, binds, env=env)
    # run_dir is a deployment-controlled scratch path (space-free), so the
    # JUnit path is emitted unquoted to let ${SLURM_JOB_ID} expand.
    junit = f"{run_dir}/{junit_label}_${{SLURM_JOB_ID}}.xml"
    return (
        prefix
        + " python -m pytest --pyargs picasso_workflow"
        + f" -m {shlex.quote(marker)}"
        + f" --junit-xml={junit} -v"
    )


def tiers1_3_commands(image_path, run_dir, module, bind_paths):
    """Commands for the tiers-1-3 job: unit+template gate, then integration.

    The unit/template tier gates the integration tier (the integration pytest
    only runs if the first exits 0), exactly like the old ``afterok`` inside
    one allocation. ``PW_RC`` carries the job's exit status (``create_slurm_script``
    appends ``exit ${PW_RC:-0}``).
    """
    return [
        "source /etc/profile.d/modules.sh",
        f"module load {shlex.quote(module)}",
        pytest_command(
            image_path, run_dir, "not integration", "tier1_2", bind_paths
        ),
        "PW_RC=$?",
        'if [ "$PW_RC" -eq 0 ]; then',
        "    "
        + pytest_command(
            image_path,
            run_dir,
            "integration and not real_data",
            "tier3",
            bind_paths,
        ),
        "    PW_RC=$?",
        "fi",
    ]


def tier4_commands(image_path, run_dir, module, bind_paths, test_data_dir):
    """Commands for the tier-4 job: real-data integration tests."""
    return [
        "source /etc/profile.d/modules.sh",
        f"module load {shlex.quote(module)}",
        pytest_command(
            image_path,
            run_dir,
            "integration and real_data",
            "tier4",
            bind_paths,
            test_data_dir=test_data_dir,
        ),
        "PW_RC=$?",
    ]


def summary_commands(run_dir):
    """Commands for the summary job: run the staged summarizer on the node.

    Runs with the node's ``python3`` (NOT inside the container) because it
    shells out to ``sacct``, which the image has no client for. The summarizer
    is stdlib-only, so a bare ``python3 <file> <run_dir>`` suffices.
    """
    summarize = posixpath.join(run_dir, "summarize.py")
    return [
        f"python3 {shlex.quote(summarize)} {shlex.quote(run_dir)}",
        "PW_RC=$?",
    ]


def render_jobs_tsv(rows):
    """Render the ``jobs.tsv`` manifest the summarizer reads.

    Each row is ``(kind, jobid, label, dir)``; ``dir`` is ``-`` for tiers.
    """
    return "".join("\t".join(str(c) for c in row) + "\n" for row in rows)


def render_run_info(info):
    """Render the human-readable ``run_info.txt`` from an ordered mapping."""
    return "".join(f"{k}: {v}\n" for k, v in info.items())


# ---------------------------------------------------------------------------
# SSH orchestration
# ---------------------------------------------------------------------------
class TestChainSubmitter:
    """Stage + submit the container test chain over SSH.

    Built on a :class:`picasso_workflow.admin.core.ClusterAdmin` (its
    ``SlurmCommunicator`` does the scp/sbatch). ``summarize_source`` is the
    stdlib-only summarizer's text, staged into the run dir.
    """

    def __init__(
        self,
        admin,
        image_path,
        test_results_dir,
        module,
        bind_paths,
        summarize_source,
    ):
        self.admin = admin
        self.comm = admin.comm
        self.image_path = image_path
        self.test_results_dir = test_results_dir
        self.module = module
        self.bind_paths = bind_paths
        self.summarize_source = summarize_source

    def _sbatch(self, label, commands, resources, run_dir, dependency=None):
        """Stage one sbatch script and submit it; return its job id (str)."""
        script = self.comm.create_slurm_script(
            f"pw-test-{label}",
            commands,
            slurm_options=resources,
            output_file=f"{run_dir}/{label}_%j.log",
            error_file=f"{run_dir}/{label}_%j.log",
            working_directory=run_dir,
        )
        remote = f"{run_dir}/{label}.sbatch"
        res = self.comm.put_file(script, remote, make_executable=True)
        if not res.get("success"):
            raise RuntimeError(
                f"staging {remote} failed: "
                f"{(res.get('stderr', '') or '').strip()}"
            )
        return self._submit(remote, dependency)

    def _submit(self, remote_script, dependency=None):
        cmd = "sbatch --parsable"
        if dependency:
            cmd += f" --dependency={dependency}"
        cmd += f" {shlex.quote(remote_script)}"
        # --parsable prints just "<jobid>" or "<jobid>;<cluster>".
        out = self.admin._run(cmd)
        return out.strip().split(";")[0].split()[0]

    def discover_templates(self, test_data_dir):
        """Template run scripts under ``test_data_dir`` (sorted)."""
        out = self.admin._run(
            f"find {shlex.quote(test_data_dir)} -type f "
            "-name run_workflow_slurm.sh"
        )
        return sorted(p.strip() for p in out.splitlines() if p.strip())

    def submit(
        self,
        version,
        timestamp,
        test_data_dir=None,
        run_tier4=True,
        run_templates=True,
    ):
        """Stage + submit the whole chain; return a summary dict.

        Returns ``{run_dir, tiers_job, tier4_job, template_jobs, summary_job}``.
        """
        if not self.test_results_dir:
            raise RuntimeError(
                "ClusterApptainer.test_results_dir is not set in config.yaml."
            )
        run_dir = run_dir_path(self.test_results_dir, version, timestamp)
        # run dir + the writable cache dirs the in-image pytest needs.
        self.admin._run(
            "mkdir -p "
            + shlex.quote(run_dir)
            + " "
            + shlex.quote(f"{run_dir}/.numba")
            + " "
            + shlex.quote(f"{run_dir}/.mpl")
        )
        # stage the stdlib-only summarizer for the (on-node) summary job.
        self.comm.put_file(self.summarize_source, f"{run_dir}/summarize.py")

        tiers_job = self._sbatch(
            "tiers1_3",
            tiers1_3_commands(
                self.image_path, run_dir, self.module, self.bind_paths
            ),
            _RES_TIERS,
            run_dir,
        )

        rows = [
            ("tier", tiers_job, "tier1_2", "-"),
            ("tier", tiers_job, "tier3", "-"),
        ]
        dep_ids = [tiers_job]

        tier4_job = None
        if run_tier4 and test_data_dir:
            tier4_job = self._sbatch(
                "tier4",
                tier4_commands(
                    self.image_path,
                    run_dir,
                    self.module,
                    self.bind_paths,
                    test_data_dir,
                ),
                _RES_TIER4,
                run_dir,
                dependency=f"afterok:{tiers_job}",
            )
            rows.append(("tier", tier4_job, "tier4", "-"))
            dep_ids.append(tier4_job)

        template_jobs = []
        if run_templates and test_data_dir:
            for script in self.discover_templates(test_data_dir):
                jid = self._submit(script, dependency=f"afterok:{tiers_job}")
                template_jobs.append((jid, script))
                rows.append(
                    (
                        "template",
                        jid,
                        posixpath.basename(posixpath.dirname(script)),
                        script,
                    )
                )
                dep_ids.append(jid)

        # manifest + run info, then the summary job (afterany everything).
        self.comm.put_file(render_jobs_tsv(rows), f"{run_dir}/jobs.tsv")
        self.comm.put_file(
            render_run_info(
                {
                    "run_id": posixpath.basename(run_dir),
                    "version": version or "dev",
                    "image": self.image_path,
                    "test_data_dir": test_data_dir or "<none>",
                    "tiers1_3_job": tiers_job,
                    "tier4_job": tier4_job or "<skipped>",
                    "templates": len(template_jobs),
                }
            ),
            f"{run_dir}/run_info.txt",
        )
        summary_job = self._sbatch(
            "summary",
            summary_commands(run_dir),
            _RES_SUMMARY,
            run_dir,
            dependency="afterany:" + ":".join(dep_ids),
        )

        return {
            "run_dir": run_dir,
            "tiers_job": tiers_job,
            "tier4_job": tier4_job,
            "template_jobs": template_jobs,
            "summary_job": summary_job,
        }
