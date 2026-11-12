#!/usr/bin/env python
"""Unit tests for the container test-chain orchestration (Qt-free, no SSH).

The command/script builders in :mod:`picasso_workflow.admin.testrun` are pure,
so the risky assembly (apptainer exec with binds + ``--env`` cache dirs +
explicit ``-m`` markers, the tier gate, the dependency chain) is tested without
a cluster. ``TestChainSubmitter`` is exercised against a fake admin that records
the SSH commands.
"""

from __future__ import annotations

from picasso_workflow.admin import testrun


# --------------------------------------------------------------------------- #
# pure builders
# --------------------------------------------------------------------------- #
def test_run_dir_path():
    assert (
        testrun.run_dir_path("/scratch/tr", "0.9.0", "20261112_101500")
        == "/scratch/tr/20261112_101500_0.9.0"
    )
    assert testrun.run_dir_path("/scratch/tr", None, "T").endswith("_dev")


def test_cache_env():
    env = testrun.cache_env("/scratch/run")
    assert env["NUMBA_CACHE_DIR"] == "/scratch/run/.numba"
    assert env["MPLCONFIGDIR"] == "/scratch/run/.mpl"


def test_pytest_command_tiers():
    cmd = testrun.pytest_command(
        "/sw/x-test.sif",
        "/scratch/run",
        "not integration",
        "tier1_2",
        ["/scratch/paint"],
    )
    assert cmd.startswith("apptainer exec ")
    assert "--env NUMBA_CACHE_DIR=/scratch/run/.numba" in cmd
    assert "--bind /scratch/run" in cmd
    assert "/sw/x-test.sif " in cmd
    assert "python -m pytest --pyargs picasso_workflow" in cmd
    assert "-m 'not integration'" in cmd
    # JUnit path left unquoted so the node shell expands ${SLURM_JOB_ID}
    assert "--junit-xml=/scratch/run/tier1_2_${SLURM_JOB_ID}.xml" in cmd
    # no real-data env unless a data dir is given
    assert "PW_TEST_DATA_DIR" not in cmd


def test_pytest_command_tier4_binds_and_exports_data_dir():
    cmd = testrun.pytest_command(
        "/sw/x-test.sif",
        "/scratch/run",
        "integration and real_data",
        "tier4",
        ["/scratch/paint"],
        test_data_dir="/fs/testdata",
    )
    assert "--env PW_TEST_DATA_DIR=/fs/testdata" in cmd
    assert "--bind /fs/testdata" in cmd
    assert "-m 'integration and real_data'" in cmd


def test_tiers1_3_commands_gate():
    cmds = testrun.tiers1_3_commands(
        "/sw/x-test.sif", "/scratch/run", "apptainer/1.4.1", ["/scratch/paint"]
    )
    assert "module load apptainer/1.4.1" in cmds
    # unit/template tier, then a gated integration tier
    assert any("-m 'not integration'" in c for c in cmds)
    assert 'if [ "$PW_RC" -eq 0 ]; then' in cmds
    assert any("-m 'integration and not real_data'" in c for c in cmds)
    assert "fi" in cmds
    assert cmds.count("PW_RC=$?") >= 1


def test_summary_commands_runs_on_node():
    cmds = testrun.summary_commands("/scratch/run")
    # NOT inside apptainer (needs sacct, absent from the image)
    assert not any("apptainer" in c for c in cmds)
    assert "python3 /scratch/run/summarize.py /scratch/run" in cmds[0]


def test_render_jobs_tsv_and_run_info():
    tsv = testrun.render_jobs_tsv(
        [("tier", "101", "tier1_2", "-"), ("template", "105", "t1", "/d/t1")]
    )
    assert tsv == "tier\t101\ttier1_2\t-\ntemplate\t105\tt1\t/d/t1\n"
    info = testrun.render_run_info({"run_id": "r", "version": "0.9.0"})
    assert info == "run_id: r\nversion: 0.9.0\n"


# --------------------------------------------------------------------------- #
# orchestration (fake admin: records SSH commands, hands out job ids)
# --------------------------------------------------------------------------- #
class _FakeComm:
    def __init__(self):
        self.put = []

    def create_slurm_script(self, name, commands, **kw):
        return f"#script:{name}\n" + "\n".join(commands)

    def put_file(self, content, remote, make_executable=False):
        self.put.append(remote)
        return {"success": True, "stdout": "", "stderr": "", "return_code": 0}


class _FakeAdmin:
    def __init__(self, templates):
        self.host = "clusterXXX"
        self.comm = _FakeComm()
        self.templates = templates
        self.commands = []
        self._next = 100

    def _run(self, command):
        self.commands.append(command)
        if command.startswith("mkdir"):
            return ""
        if command.startswith("find "):
            return "\n".join(self.templates) + "\n"
        if command.startswith("sbatch"):
            self._next += 1
            return f"{self._next}\n"
        return ""


def _submitter(admin):
    return testrun.TestChainSubmitter(
        admin,
        image_path="/sw/picasso-workflow-test-0.9.0.sif",
        test_results_dir="/scratch/tr",
        module="apptainer/1.4.1",
        bind_paths=["/scratch/paint"],
        summarize_source="# summarize\n",
    )


def test_submit_full_chain_dependencies():
    admin = _FakeAdmin(templates=["/fs/testdata/t1/run_workflow_slurm.sh"])
    sub = _submitter(admin)
    result = sub.submit(
        version="0.9.0",
        timestamp="20261112_101500",
        test_data_dir="/fs/testdata",
    )
    assert result["run_dir"] == "/scratch/tr/20261112_101500_0.9.0"
    assert result["tiers_job"] == "101"
    assert result["tier4_job"] == "102"
    assert len(result["template_jobs"]) == 1
    assert result["summary_job"] is not None

    sbatches = [c for c in admin.commands if c.startswith("sbatch")]
    # tiers first (no dependency), then tier4/template afterok tiers, summary
    # afterany on all.
    assert "--dependency" not in sbatches[0]
    assert "--dependency=afterok:101" in sbatches[1]  # tier4
    assert "--dependency=afterok:101" in sbatches[2]  # template
    assert sbatches[-1].count("afterany:101:102:") == 1  # summary on all
    # staged artefacts: summarizer + manifest + run info
    assert "/scratch/tr/20261112_101500_0.9.0/summarize.py" in admin.comm.put
    assert "/scratch/tr/20261112_101500_0.9.0/jobs.tsv" in admin.comm.put


def test_submit_tiers_only_when_no_data_dir():
    admin = _FakeAdmin(templates=[])
    result = _submitter(admin).submit(
        version="0.9.0", timestamp="T", test_data_dir=None
    )
    assert result["tier4_job"] is None
    assert result["template_jobs"] == []
    # only tiers + summary submitted
    sbatches = [c for c in admin.commands if c.startswith("sbatch")]
    assert len(sbatches) == 2
    assert "afterany:101" in sbatches[-1]
