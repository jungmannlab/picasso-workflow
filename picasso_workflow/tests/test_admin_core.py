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
    assert lean.mtime == "2026-11-12 10:30"
    assert lean.size.endswith("G")  # ~4.1 GB -> human readable
    assert images[2].is_test is True


def test_parse_image_listing_empty():
    assert core.parse_image_listing("total 0\n") == []


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
