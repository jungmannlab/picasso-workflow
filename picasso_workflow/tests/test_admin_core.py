#!/usr/bin/env python
"""Unit tests for the admin console's pure output parsers.

The admin console inspects the cluster over SSH; the command-output parsers in
:mod:`picasso_workflow.admin.core` are pure functions, so they test without a
cluster (mirrors monet's ``test_admin_core``). No Qt, no SSH.
"""

from __future__ import annotations

from picasso_workflow.admin import core


# --------------------------------------------------------------------------- #
# image-name / version helpers
# --------------------------------------------------------------------------- #
def test_parse_image_name_lean_and_test():
    assert core.parse_image_name("picasso-workflow-0.9.0.sif") == (
        "0.9.0",
        False,
    )
    assert core.parse_image_name("picasso-workflow-test-0.9.0.sif") == (
        "0.9.0",
        True,
    )


def test_parse_image_name_develop_unparsed():
    # develop-<sha> images don't map to a release version, but the test marker
    # is still detected.
    assert core.parse_image_name("picasso-workflow-develop-a1b2c3d.sif") == (
        None,
        False,
    )
    assert core.parse_image_name(
        "picasso-workflow-test-develop-a1b2c3d.sif"
    ) == (None, True)


def test_parse_image_name_malformed_version_unparsed():
    # dotted-but-not-a-version names are shown verbatim (version=None), not
    # mislabelled as a parsed version.
    assert core.parse_image_name("picasso-workflow-0..0.sif") == (None, False)
    assert core.parse_image_name("picasso-workflow-.sif") == (None, False)


def test_default_test_image_basename():
    assert (
        core.default_test_image_basename("0.9.0")
        == "picasso-workflow-test-0.9.0.sif"
    )
    assert core.default_test_image_basename("unknown") is None
    assert core.default_test_image_basename("0.9.1.dev3+g1a2b") is None


# --------------------------------------------------------------------------- #
# ls -l --time-style=long-iso
# --------------------------------------------------------------------------- #
def test_parse_image_listing():
    out = (
        "total 8389000\n"
        "-rw-r--r-- 1 u grp 4100000000 2026-11-12 10:30 "
        "picasso-workflow-0.9.0.sif\n"
        "-rw-r--r-- 1 u grp 4300000000 2026-11-11 09:00 "
        "picasso-workflow-test-0.9.0.sif\n"
        "lrwxrwxrwx 1 u grp 26 2026-11-12 10:31 "
        "picasso-workflow-latest.sif -> picasso-workflow-0.9.0.sif\n"
        "drwxr-xr-x 2 u grp 4096 2026-11-10 08:00 subdir\n"
        "-rw-r--r-- 1 u grp 123 2026-11-10 08:00 notes.txt\n"
    )
    images = core.parse_image_listing(out)
    names = [i.name for i in images]
    # .sif regular files AND symlinks (name without the '-> target'), sorted
    assert names == [
        "picasso-workflow-0.9.0.sif",
        "picasso-workflow-latest.sif",
        "picasso-workflow-test-0.9.0.sif",
    ]
    lean = images[0]
    assert lean.version == "0.9.0" and lean.is_test is False
    assert lean.tag == "0.9.0"
    assert lean.mtime == "2026-11-12 10:30"
    assert lean.size.endswith("G")  # ~4.1 GB -> human readable
    assert images[1].tag == "latest"  # symlink alias shows its label
    assert images[2].is_test is True and images[2].tag == "0.9.0"


def test_parse_image_listing_empty():
    assert core.parse_image_listing("total 0\n") == []


def test_configured_test_data_dir():
    assert (
        core.configured_test_data_dir({"TestData": {"directory": "/fs/td"}})
        == "/fs/td"
    )
    # unset / null / missing -> "" (blank => tiers 1-3 only)
    assert (
        core.configured_test_data_dir({"TestData": {"directory": None}}) == ""
    )
    assert core.configured_test_data_dir({}) == ""
    # a malformed (non-dict) TestData must NOT crash -> "" (GUI startup safety)
    assert core.configured_test_data_dir({"TestData": "/fs/td"}) == ""
    assert core.configured_test_data_dir({"TestData": ["/fs/td"]}) == ""


def test_configured_test_data_dir_default_config():
    # the GUI calls it with no args -> reads the module CONFIG (the real call
    # site); mutate+restore the shared dict to exercise that branch.
    saved = dict(core.CONFIG)
    try:
        core.CONFIG.clear()
        core.CONFIG["TestData"] = {"directory": "/fs/from-config"}
        assert core.configured_test_data_dir() == "/fs/from-config"
    finally:
        core.CONFIG.clear()
        core.CONFIG.update(saved)


def test_image_tag():
    # release / test both show the version; a develop image shows its label
    assert core.image_tag("picasso-workflow-0.9.0.sif") == "0.9.0"
    assert core.image_tag("picasso-workflow-test-0.9.0.sif") == "0.9.0"
    assert (
        core.image_tag("picasso-workflow-develop-ef8b42.sif")
        == "develop-ef8b42"
    )
    # a non-picasso-workflow name: tag is the whole stem, and it is NOT
    # mis-parsed as a version even if it is all-digits.
    assert core.image_tag("123.sif") == "123"
    assert core.parse_image_name("123.sif") == (None, False)


# --------------------------------------------------------------------------- #
# squeue / sacct
# --------------------------------------------------------------------------- #
def test_parse_squeue():
    t = core._SEP
    out = (
        t.join(["12345", "pwjob", "RUNNING", "1:02:03", "gpu", "(None)"])
        + "\n"
        + t.join(["12346", "pw|job2", "PENDING", "0:00", "cpu", "(Resources)"])
        + "\n"
    )
    jobs = core.parse_squeue(out)
    assert [j.job_id for j in jobs] == ["12345", "12346"]
    assert jobs[0].state == "RUNNING" and jobs[0].elapsed == "1:02:03"
    assert "gpu" in jobs[0].extra and "(None)" in jobs[0].extra
    # a pipe in the job name must NOT shift fields (tab-delimited)
    assert jobs[1].name == "pw|job2" and jobs[1].state == "PENDING"


def test_parse_sacct():
    t = core._SEP
    out = (
        t.join(["12345", "pwjob", "COMPLETED", "00:42:10", "2026-11-12T11:12"])
        + "\n"
        + t.join(["12346", "pwjob2", "FAILED", "00:01:00", "2026-11-12T11:20"])
        + "\n"
    )
    jobs = core.parse_sacct(out)
    assert [j.state for j in jobs] == ["COMPLETED", "FAILED"]
    assert jobs[0].elapsed == "00:42:10"
    assert jobs[0].extra == "2026-11-12T11:12"


def test_parse_squeue_sacct_empty():
    assert core.parse_squeue("") == []
    assert core.parse_sacct("\n") == []


# --------------------------------------------------------------------------- #
# df
# --------------------------------------------------------------------------- #
def test_parse_df():
    out = (
        "Filesystem      Size  Used Avail Use% Mounted on\n"
        "/dev/sdb1       2.0T  812G  1.2T  41% /sw\n"
    )
    du = core.parse_df(out)
    assert du is not None
    assert du.size == "2.0T" and du.used == "812G"
    assert du.avail == "1.2T" and du.use_pct == "41%"
    assert du.path == "/sw"


def test_parse_df_no_data():
    assert (
        core.parse_df("Filesystem Size Used Avail Use% Mounted on\n") is None
    )


# --------------------------------------------------------------------------- #
# cancel_jobs / fetch_logs (SSH ops against a recording fake comm)
# --------------------------------------------------------------------------- #
class _FakeComm:
    """Records SSH commands; replies success with a canned stdout."""

    def __init__(self, stdout=""):
        self.commands = []
        self.stdout = stdout

    def execute_ssh_command(self, command):
        self.commands.append(command)
        return {
            "success": True,
            "stdout": self.stdout,
            "stderr": "",
            "return_code": 0,
        }


def _admin(stdout=""):
    return core.ClusterAdmin(_FakeComm(stdout), "clusterXXX", "u")


def test_cancel_jobs_one_scancel_for_all():
    admin = _admin()
    assert admin.cancel_jobs(["101", "102", "103"]) == 3
    # a single scancel lists every id (not one call per job)
    scancels = [c for c in admin.comm.commands if c.startswith("scancel ")]
    assert len(scancels) == 1
    assert "101" in scancels[0] and "103" in scancels[0]


def test_cancel_jobs_filters_none_and_placeholder():
    # skipped tiers hand us None / '-'; those must never reach scancel.
    admin = _admin()
    assert admin.cancel_jobs([None, "-", "101"]) == 1
    assert admin.comm.commands == ["scancel 101"]


def test_cancel_jobs_noop_when_empty():
    admin = _admin()
    assert admin.cancel_jobs([None, "-"]) == 0
    # nothing to cancel => no SSH at all
    assert admin.comm.commands == []


def test_fetch_logs_tails_run_dir_logs():
    admin = _admin(stdout="===== /run/tiers1_3_101.log =====\nFAILED\n")
    out = admin.fetch_logs("/scratch/run", tail=50)
    assert "FAILED" in out
    (cmd,) = admin.comm.commands
    # globs the run dir's *.log, guards each entry, tails it; an unreadable
    # single log is swallowed so it doesn't look like an SSH failure
    assert "/scratch/run/*.log" in cmd
    assert "tail -n 50" in cmd
    assert '[ -e "$f" ]' in cmd
    assert "|| true" in cmd


def test_fetch_logs_empty_when_none():
    admin = _admin(stdout="")
    assert admin.fetch_logs("/scratch/run") == ""


# --------------------------------------------------------------------------- #
# submit_test_chain: Drivepaths mapping of the test-data dir
# --------------------------------------------------------------------------- #
class _SubmitComm:
    """Fake comm for submit_test_chain: hands out job ids, records commands."""

    def __init__(self):
        self.commands = []
        self._next = 100

    def execute_ssh_command(self, command):
        self.commands.append(command)
        stdout = ""
        if command.startswith("sbatch"):
            self._next += 1
            stdout = f"{self._next}\n"
        return {
            "success": True,
            "stdout": stdout,
            "stderr": "",
            "return_code": 0,
        }

    def put_file(self, content, remote, make_executable=False):
        return {"success": True, "stdout": "", "stderr": "", "return_code": 0}

    def create_slurm_script(self, name, commands, **kw):
        return "#script\n"


def test_submit_test_chain_maps_test_data_dir_via_drivepaths(monkeypatch):
    # A local "/Volumes/..." test-data dir must be rewritten to the cluster
    # host's "/fs/mpib/..." layout (via Drivepaths) before the remote `find`
    # and the apptainer binds use it -- otherwise it is a no-such-dir on the
    # login node.
    monkeypatch.setitem(
        core.CONFIG,
        "Drivepaths",
        {
            "pcjuXX.local": ["/Volumes/pool-miblab1"],
            "hpcl8XXX": ["/fs/mpib/pool-miblab1"],
        },
    )
    monkeypatch.setattr(
        core, "apptainer_config", lambda host: ("/sw", "apptainer/1.4.1", [])
    )
    monkeypatch.setattr(
        core, "apptainer_test_results_dir", lambda host: "/scratch/tr"
    )

    admin = core.ClusterAdmin(_SubmitComm(), "hpcl8XXX", "u")
    result = admin.submit_test_chain(
        image_path="/sw/x-test.sif",
        version="0.9.0",
        timestamp="T",
        test_data_dir="/Volumes/pool-miblab1/templates",
        run_tier4=False,
    )
    # the returned (converted) dir and the remote `find` both use the cluster
    # layout, never the local /Volumes path.
    assert result["test_data_dir"] == "/fs/mpib/pool-miblab1/templates"
    finds = [c for c in admin.comm.commands if c.startswith("find ")]
    assert len(finds) == 1
    assert "/fs/mpib/pool-miblab1/templates" in finds[0]
    assert "/Volumes" not in finds[0]


def test_submit_test_chain_path_beside_mapped_root_passes_through(monkeypatch):
    # A dir BESIDE a mapped root (".../pool-miblab1-extra" next to the
    # ".../pool-miblab1" root) is NOT under it, so conversion leaves it
    # unchanged -- and the submit proceeds with the entered path rather than
    # crashing (this exercised the old substring-match crash; now a clean
    # no-op). See test_util for the metaworkflow-level coverage.
    monkeypatch.setitem(
        core.CONFIG,
        "Drivepaths",
        {
            "pcjuXX.local": ["/Volumes/pool-miblab1"],
            "hpcl8XXX": ["/fs/mpib/pool-miblab1"],
        },
    )
    monkeypatch.setattr(
        core, "apptainer_config", lambda host: ("/sw", "apptainer/1.4.1", [])
    )
    monkeypatch.setattr(
        core, "apptainer_test_results_dir", lambda host: "/scratch/tr"
    )

    admin = core.ClusterAdmin(_SubmitComm(), "hpcl8XXX", "u")
    result = admin.submit_test_chain(
        image_path="/sw/x-test.sif",
        version="0.9.0",
        timestamp="T",
        test_data_dir="/Volumes/pool-miblab1-extra/templates",
        run_tier4=False,
    )
    # beside-root -> not under any root -> left unchanged, submit proceeded
    assert result["test_data_dir"] == "/Volumes/pool-miblab1-extra/templates"
    finds = [c for c in admin.comm.commands if c.startswith("find ")]
    assert len(finds) == 1
    assert "/Volumes/pool-miblab1-extra/templates" in finds[0]


def test_submit_test_chain_guards_conversion_exception(monkeypatch):
    # The call-site try/except is defence-in-depth: if the shared converter
    # ever raises, the submit must fall back to the entered path, not abort
    # with a traceback at the GUI boundary.
    import picasso_workflow.util as pwutil

    def _boom(path, dest_machine=None):
        raise RuntimeError("converter blew up")

    monkeypatch.setattr(pwutil, "convert_filepath_for_machine", _boom)
    monkeypatch.setattr(
        core, "apptainer_config", lambda host: ("/sw", "apptainer/1.4.1", [])
    )
    monkeypatch.setattr(
        core, "apptainer_test_results_dir", lambda host: "/scratch/tr"
    )

    admin = core.ClusterAdmin(_SubmitComm(), "hpcl8XXX", "u")
    result = admin.submit_test_chain(
        image_path="/sw/x-test.sif",
        version="0.9.0",
        timestamp="T",
        test_data_dir="/Volumes/pool-miblab1/templates",
        run_tier4=False,
    )
    # converter raised -> entered path kept, no exception propagated
    assert result["test_data_dir"] == "/Volumes/pool-miblab1/templates"
    finds = [c for c in admin.comm.commands if c.startswith("find ")]
    assert len(finds) == 1
    assert "/Volumes/pool-miblab1/templates" in finds[0]
