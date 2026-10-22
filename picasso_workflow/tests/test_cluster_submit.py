#!/usr/bin/env python
"""WP-9 Phase C cluster-submit worker tests (hermetic).

Exercises the ``picasso_workflow.cluster_submit`` worker end-to-end against the
in-memory ``picasso_registry.testing.mock_registry`` (FastAPI TestClient over
in-memory SQLite) with a MOCK SSH communicator + a no-network stager -- no real
cluster, no network. Asserts the claim->submit happy path (exactly-once claim,
analysis_run created with Phase-3 ``extra.template_conformance``, hand-off
claimed->running with analysis_run_id, one sbatch submission with apptainer
module-load + undrift-first), the nothing-ready no-op, the write-back contract
(running->done / running->failed, illegal-transition 409), and the
submit-failure "never lose a dataset" behaviour (hand-off -> failed, locs
pointers preserved).

Registry-backed tests carry the ``@requires_registry`` skip (picasso-registry's
client stack is NOT a picasso-workflow dep), so they skip on the hosted
wheel-only CI and run in the dev container.
"""

from __future__ import annotations

import unittest

from picasso_workflow import cluster_submit as cs
from picasso_workflow.cluster_submit import (
    ClusterSubmitConfig,
    ClusterSubmitWorker,
    build_cluster_workflow,
    build_sbatch_script,
    cluster_writeback,
)

# picasso-registry client stack (mock_registry) is not a base/dev dep — these
# tests skip on the hosted wheel-only CI and run in the full dev container.
try:
    from picasso_registry.testing import mock_registry as _mock_registry

    _ = _mock_registry
    HAS_REGISTRY = True
except Exception:
    HAS_REGISTRY = False

requires_registry = unittest.skipUnless(
    HAS_REGISTRY,
    "picasso-registry client stack (picasso-registry[client]) not installed "
    "(hosted wheel-only CI)",
)


class MockSlurmCommunicator:
    """A no-network ``SlurmCommunicator`` double.

    Records every SSH command + sbatch submission so tests assert on the
    script content (apptainer module-load, undrift-first) and submission
    count without touching the network. ``submit_fails`` flips sbatch to a
    failed result to exercise the don't-lose-a-dataset path.
    """

    def __init__(self, submit_fails: bool = False):
        self.ssh_commands: list[str] = []
        self.submitted: list[tuple[str, str]] = []
        self.submit_fails = submit_fails

    def execute_ssh_command(self, command, working_directory=None):
        self.ssh_commands.append(command)
        return {"stdout": "", "stderr": "", "return_code": 0, "success": True}

    def submit_job(self, script_path, dest_machine, additional_options=None):
        # Capture the script we wrote to scratch via the heredoc SSH write.
        script = self._last_written_script(script_path)
        self.submitted.append((script_path, script))
        if self.submit_fails:
            return {
                "job_id": None,
                "stdout": "",
                "stderr": "sbatch: error: invalid partition",
                "success": False,
            }
        return {
            "job_id": 4242,
            "stdout": "Submitted batch job 4242",
            "stderr": "",
            "success": True,
        }

    def _last_written_script(self, script_path: str) -> str:
        # The worker writes the script via `cat > <path> <<'PWF_EOF' ...`.
        for cmd in reversed(self.ssh_commands):
            if cmd.startswith(f"cat > {script_path} "):
                body = cmd.split("\n", 1)[1]
                return body.rsplit("\nPWF_EOF", 1)[0]
        return ""


def _fake_stager(local_paths, remote_dir, config):
    """No-network stager: pretend the locs landed under ``remote_dir``."""
    return [f"{remote_dir}/{p.rsplit('/', 1)[-1]}" for p in local_paths]


def _ready_dataset(reg, acq_id="acq-1", locs=None):
    """Create an acquisition_run and ready it for hand-off (QC's mark)."""
    reg.log_acquisition(id=acq_id, status="done")
    locs = locs or [
        f"/data/{acq_id}/channel 0.hdf5",
        f"/data/{acq_id}/channel 1.hdf5",
    ]
    reg.mark_ready(acq_id, detail={"locs": locs})
    return acq_id, locs


# ---------------------------------------------------------------------------
# Pure (no-registry) tests — always run.
# ---------------------------------------------------------------------------
class TestWorkflowAndScript(unittest.TestCase):
    def test_cluster_workflow_is_undrift_first(self):
        wf = build_cluster_workflow(["/s/c0.hdf5", "/s/c1.hdf5"])
        names = [m for m, _ in wf]
        # The first real processing module (after loading locs) is undrift.
        self.assertEqual(names[0], "load_dataset_localizations")
        self.assertEqual(names[1], "undrift_aim")
        self.assertIn("load_datasets_to_aggregate", names)

    def test_cluster_workflow_rcc_option(self):
        wf = build_cluster_workflow(["/s/c0.hdf5"], undrift="rcc")
        self.assertEqual(wf[1][0], "undrift_rcc")

    def test_sbatch_apptainer_module_load_pinned_and_undrift_first(self):
        script = build_sbatch_script(
            job_name="j",
            workflow_yaml_remote="/scratch/acq-1/workflow.yaml",
            results_dir_remote="/scratch/acq-1",
            image_path="/sw/pwf.sif",
            partition="gpu",
            constraint="gpu",
        )
        # (a) module-load apptainer, pinned to the lowest node-group version.
        self.assertIn("module load apptainer/1.4.1", script)
        # module load precedes the apptainer exec.
        self.assertLess(
            script.index("module load apptainer"),
            script.index("apptainer exec"),
        )
        # (b) runs the aggregation inside the image; node-group constrained.
        self.assertIn("apptainer exec", script)
        self.assertIn("--partition=gpu", script)
        self.assertIn("--constraint=gpu", script)
        # exit code propagated.
        self.assertIn("exit ${PW_RC:-0}", script)

    def test_sbatch_quotes_paths_with_spaces(self):
        # Lab data folders contain spaces; the apptainer argv and #SBATCH
        # directives must survive word-splitting (regression: unquoted paths
        # split --spec <half-a-path>).
        import shlex

        script = build_sbatch_script(
            job_name="pwf-agg-my run 1",
            workflow_yaml_remote="/scratch/my run 1/workflow.yaml",
            results_dir_remote="/scratch/my run 1",
            image_path="/sw/picasso workflow.sif",
        )
        exec_line = next(
            ln for ln in script.splitlines() if ln.startswith("apptainer exec")
        )
        toks = shlex.split(exec_line)  # the real shell tokenization
        self.assertEqual(
            toks[toks.index("--spec") + 1], "/scratch/my run 1/workflow.yaml"
        )
        self.assertEqual(
            toks[toks.index("--results-dir") + 1], "/scratch/my run 1"
        )
        self.assertEqual(toks[toks.index("--bind") + 1], "/scratch/my run 1")
        # the image path is a single token (not split on its space).
        self.assertIn("/sw/picasso workflow.sif", toks)
        # the #SBATCH output pattern is quoted (one directive, not split).
        out_line = next(ln for ln in script.splitlines() if "--output=" in ln)
        self.assertIn("'/scratch/my run 1/", out_line)


# ---------------------------------------------------------------------------
# Registry-backed tests — skip on hosted CI.
# ---------------------------------------------------------------------------
@requires_registry
class TestClusterSubmitHappyPath(unittest.TestCase):
    def test_claim_submit_creates_analysis_run_and_transitions(self):
        with _mock_registry() as reg:
            acq_id, locs = _ready_dataset(reg)
            comm = MockSlurmCommunicator()
            worker = ClusterSubmitWorker(
                reg,
                ClusterSubmitConfig(partition="gpu", constraint="gpu"),
                communicator=comm,
                stager=_fake_stager,
                worker_id="node-a@1",
            )

            summary = worker.run_once()

            self.assertIsNotNone(summary)
            self.assertTrue(summary["submitted"])
            self.assertEqual(summary["job_id"], 4242)
            self.assertEqual(summary["conformance"], "exact")

            # analysis_run created with Phase-3 conformance in extra.
            ar = reg.get("analysis_run", summary["analysis_run_id"])
            self.assertEqual(ar["kind"], "cluster-aggregation")
            self.assertEqual(ar["compute_location"], "cluster")
            conf = ar["extra"]["template_conformance"]
            self.assertEqual(conf["template_id"], "cluster_aggregation")
            self.assertEqual(conf["conformance"], "exact")
            self.assertIsNotNone(conf["template_version"])

            # hand-off claimed -> running, analysis_run_id set.
            ho = [
                h
                for h in reg.list_handoffs()
                if h["acquisition_run_id"] == acq_id
            ][0]
            self.assertEqual(ho["state"], "running")
            self.assertEqual(ho["analysis_run_id"], summary["analysis_run_id"])

            # exactly one sbatch submission, apptainer + undrift-first.
            self.assertEqual(len(comm.submitted), 1)
            _, script = comm.submitted[0]
            self.assertIn("module load apptainer/1.4.1", script)
            self.assertIn("apptainer exec", script)
            self.assertIn(
                "python -m picasso_workflow.cluster_submit run", script
            )

            # the staged workflow.yaml holds the full aggregation spec +
            # the analysis_run id for cluster-side logging / write-back.
            import yaml as _yaml

            writes = [
                c
                for c in comm.ssh_commands
                if c.startswith("cat > ")
                and c.split("\n", 1)[0].endswith("workflow.yaml <<'PWF_EOF'")
            ]
            self.assertEqual(len(writes), 1)
            body = writes[0].split("\n", 1)[1].rsplit("\nPWF_EOF", 1)[0]
            staged = _yaml.safe_load(body)
            self.assertEqual(
                staged["analysis_run_id"], summary["analysis_run_id"]
            )
            self.assertIn("aggregation_workflow", staged)
            self.assertIn(
                "single_dataset_modules", staged["aggregation_workflow"]
            )
            self.assertEqual(
                staged["analysis_config"]["registry"]["run_id"],
                summary["analysis_run_id"],
            )

    def test_phase3_conformance_matches_record(self):
        with _mock_registry() as reg:
            acq_id, _ = _ready_dataset(reg)
            comm = MockSlurmCommunicator()
            worker = ClusterSubmitWorker(
                reg, communicator=comm, stager=_fake_stager
            )
            summary = worker.run_once()
            ar = reg.get("analysis_run", summary["analysis_run_id"])
            attached = ar["extra"]["template_conformance"]

            # Recompute the expected record independently and compare.
            from picasso_workflow import templates as tpl

            wf = build_cluster_workflow(
                [
                    f"/scratch/{acq_id}/channel 0.hdf5",
                    f"/scratch/{acq_id}/channel 1.hdf5",
                ]
            )
            expected = tpl.template_conformance_record(
                wf, tpl.get_template("cluster_aggregation")
            )
            self.assertEqual(attached["conformance"], expected["conformance"])
            self.assertEqual(attached["template_id"], expected["template_id"])
            self.assertEqual(
                attached["template_version"], expected["template_version"]
            )

    def test_exactly_once_second_worker_gets_nothing(self):
        with _mock_registry() as reg:
            _ready_dataset(reg)
            comm_a = MockSlurmCommunicator()
            comm_b = MockSlurmCommunicator()
            w_a = ClusterSubmitWorker(
                reg,
                communicator=comm_a,
                stager=_fake_stager,
                worker_id="a@1",
            )
            w_b = ClusterSubmitWorker(
                reg,
                communicator=comm_b,
                stager=_fake_stager,
                worker_id="b@1",
            )
            first = w_a.run_once()
            second = w_b.run_once()
            self.assertIsNotNone(first)
            self.assertIsNone(second)  # nothing left to claim
            self.assertEqual(len(comm_b.submitted), 0)


@requires_registry
class TestLocsResolution(unittest.TestCase):
    def test_locs_fallback_to_acquisition_extra(self):
        # A re-readied dataset loses its hand-off detail (reready resets it to
        # None), so the locs must resolve from the acquisition_run's durable
        # extra — otherwise the dataset strands in a fail/requeue loop.
        with _mock_registry() as reg:
            acq_id = "acq-fallback"
            locs = [f"/data/{acq_id}/channel 0.hdf5"]
            reg.log_acquisition(id=acq_id, status="done", extra={"locs": locs})
            # Ready with NO locs in detail (mimicking a reready that wiped it).
            reg.mark_ready(acq_id, detail=None)
            comm = MockSlurmCommunicator()
            worker = ClusterSubmitWorker(
                reg, communicator=comm, stager=_fake_stager
            )
            summary = worker.run_once()
            # Resolved from acquisition.extra → submitted, not stranded.
            self.assertTrue(summary["submitted"])
            self.assertEqual(len(comm.submitted), 1)

    def test_no_locs_anywhere_fails_cleanly(self):
        with _mock_registry() as reg:
            acq_id = "acq-nolocs"
            reg.log_acquisition(id=acq_id, status="done")
            reg.mark_ready(acq_id, detail=None)
            comm = MockSlurmCommunicator()
            worker = ClusterSubmitWorker(
                reg, communicator=comm, stager=_fake_stager
            )
            summary = worker.run_once()
            # No pointer anywhere → not submitted, hand-off marked failed
            # (never silently lost, never a crash out of run_once).
            self.assertFalse(summary["submitted"])
            self.assertEqual(len(comm.submitted), 0)
            ho = [
                h
                for h in reg.list_handoffs()
                if h["acquisition_run_id"] == acq_id
            ][0]
            self.assertEqual(ho["state"], "failed")


@requires_registry
class TestNothingReady(unittest.TestCase):
    def test_no_ready_dataset_is_clean_noop(self):
        with _mock_registry() as reg:
            comm = MockSlurmCommunicator()
            worker = ClusterSubmitWorker(
                reg, communicator=comm, stager=_fake_stager
            )
            result = worker.run_once()
            self.assertIsNone(result)
            self.assertEqual(len(comm.submitted), 0)
            self.assertEqual(len(comm.ssh_commands), 0)


@requires_registry
class TestWriteBack(unittest.TestCase):
    def _submit(self, reg, comm):
        _ready_dataset(reg)
        worker = ClusterSubmitWorker(
            reg, communicator=comm, stager=_fake_stager
        )
        return worker.run_once()

    def test_running_to_done_with_metrics(self):
        with _mock_registry() as reg:
            comm = MockSlurmCommunicator()
            summary = self._submit(reg, comm)
            row = cluster_writeback(
                reg,
                summary["acquisition_run_id"],
                success=True,
                analysis_run_id=summary["analysis_run_id"],
                metrics={"nena_nm": 4.2, "n_locs": 1000},
            )
            self.assertIsNotNone(row)
            self.assertEqual(row["state"], "done")

    def test_running_to_failed_with_detail(self):
        with _mock_registry() as reg:
            comm = MockSlurmCommunicator()
            summary = self._submit(reg, comm)
            row = cluster_writeback(
                reg,
                summary["acquisition_run_id"],
                success=False,
                analysis_run_id=summary["analysis_run_id"],
                detail={"reason": "undrift diverged"},
            )
            self.assertIsNotNone(row)
            self.assertEqual(row["state"], "failed")

    def test_illegal_transition_returns_none(self):
        with _mock_registry() as reg:
            # No running hand-off: a done write-back on a non-running dataset
            # 409s -> None (illegal edge, state not corrupted).
            acq_id, _ = _ready_dataset(reg)
            row = cluster_writeback(reg, acq_id, success=True)
            self.assertIsNone(row)  # still 'ready', not 'running'


@requires_registry
class TestSubmitFailureNeverLosesDataset(unittest.TestCase):
    def test_submit_failure_marks_failed_and_keeps_locs(self):
        with _mock_registry() as reg:
            acq_id, locs = _ready_dataset(reg)
            comm = MockSlurmCommunicator(submit_fails=True)
            worker = ClusterSubmitWorker(
                reg,
                communicator=comm,
                stager=_fake_stager,
                worker_id="node-a@1",
            )
            summary = worker.run_once()

            self.assertIsNotNone(summary)
            self.assertFalse(summary["submitted"])

            ho = [
                h
                for h in reg.list_handoffs()
                if h["acquisition_run_id"] == acq_id
            ][0]
            # Dataset landed in 'failed' (re-readyable), not lost / stuck.
            self.assertEqual(ho["state"], "failed")
            # locs pointers preserved through the failure (merged, not clobbered)
            self.assertEqual(ho["detail"]["locs"], locs)
            self.assertIn("error", ho["detail"])

            # and it can be re-queued.
            rereadied = reg.reready_handoff(acq_id)
            self.assertEqual(rereadied["state"], "ready")
            self.assertEqual(rereadied["detail"], None)


@requires_registry
class TestClusterRunEntryPoint(unittest.TestCase):
    """The cluster-side entry point drives the write-back (runner mocked).

    The real ``AggregationWorkflowRunner.run()`` is a user-side smoke test; here
    we monkeypatch the runner so ``cluster_run`` is exercised hermetically and
    we assert it closes the loop running -> done / failed.
    """

    def _staged_file(self, tmpdir, reg, success_runner):
        import os

        import picasso_workflow

        # Put a dataset into 'running' with an analysis_run to write back to.
        # Stage under the tmpdir so the staged result_location is a real,
        # writable path when cluster_run makedirs it below (on the cluster this
        # is the scratch dir; here it must be local-writable).
        acq_id, _ = _ready_dataset(reg)
        comm = MockSlurmCommunicator()
        worker = ClusterSubmitWorker(
            reg,
            ClusterSubmitConfig(scratch_root=tmpdir),
            communicator=comm,
            stager=_fake_stager,
        )
        worker.run_once()

        # Grab the staged spec yaml the worker wrote.
        body = None
        for c in comm.ssh_commands:
            if c.startswith("cat > ") and c.split("\n", 1)[0].endswith(
                "workflow.yaml <<'PWF_EOF'"
            ):
                body = c.split("\n", 1)[1].rsplit("\nPWF_EOF", 1)[0]
        spec_path = os.path.join(tmpdir, "workflow.yaml")
        with open(spec_path, "w", encoding="utf-8") as fh:
            fh.write(body)

        # Monkeypatch the runner used inside cluster_run.
        orig = picasso_workflow.AggregationWorkflowRunner

        class _FakeRunner:
            @classmethod
            def config_from_dicts(cls, **kwargs):
                return cls()

            def run(self):
                if not success_runner:
                    raise RuntimeError("boom")
                return True

        picasso_workflow.AggregationWorkflowRunner = _FakeRunner
        self.addCleanup(
            setattr, picasso_workflow, "AggregationWorkflowRunner", orig
        )
        return acq_id, spec_path

    def test_cluster_run_success_writes_back_done(self):
        import tempfile

        with _mock_registry() as reg, tempfile.TemporaryDirectory() as td:
            acq_id, spec_path = self._staged_file(td, reg, True)
            ok = cs.cluster_run(spec_path, results_dir=td, registry=reg)
            self.assertTrue(ok)
            ho = [
                h
                for h in reg.list_handoffs()
                if h["acquisition_run_id"] == acq_id
            ][0]
            self.assertEqual(ho["state"], "done")

    def test_cluster_run_failure_writes_back_failed(self):
        import tempfile

        with _mock_registry() as reg, tempfile.TemporaryDirectory() as td:
            acq_id, spec_path = self._staged_file(td, reg, False)
            ok = cs.cluster_run(spec_path, results_dir=td, registry=reg)
            self.assertFalse(ok)
            ho = [
                h
                for h in reg.list_handoffs()
                if h["acquisition_run_id"] == acq_id
            ][0]
            self.assertEqual(ho["state"], "failed")
            self.assertEqual(ho["detail"]["stage"], "cluster-run")


if __name__ == "__main__":
    unittest.main()
