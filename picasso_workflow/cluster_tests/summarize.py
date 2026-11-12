#!/usr/bin/env python3
"""Assemble one human-readable report for a cluster test run.

Reads the per-run artefacts written by ``submit_all.sh`` and condenses them
into a single ``SUMMARY.txt`` (plus a machine-readable ``summary.json``) in
the run directory.  It answers, at a glance:

  * did each pytest tier pass, and which tests failed;
  * did every *detected template workflow* run through end-to-end.

Inputs (all under ``<run_dir>``)
--------------------------------
jobs.tsv
    Tab-separated manifest written by submit_all.sh, one row per submitted
    SLURM job::

        kind <TAB> jobid <TAB> label <TAB> dir

    ``kind`` is ``tier`` or ``template``.  For tiers ``label`` is
    ``tier1_2`` / ``tier3`` / ``tier4`` and ``dir`` is ``-``.  For templates
    ``label`` is the template folder name and ``dir`` is its absolute path
    (whose ``logs/`` holds the per-job logs).

tier<label>_<jobid>.xml
    JUnit XML produced by each pytest tier.

Signals used for the verdict
----------------------------
  * ``sacct`` final State / ExitCode / Elapsed for every job id (one call).
  * JUnit XML counts for the pytest tiers (authoritative for test pass/fail).
  * For template workflows, a scan of ``<dir>/logs/*<jobid>*`` for loguru
    ERROR/CRITICAL lines and Python tracebacks, combined with the sacct
    state.  A template "ran through" iff its job COMPLETED *and* no error
    lines were found in its logs.

Usage
-----
    python summarize.py <run_dir>            # e.g. test-results/latest

Exit status is 0 when the overall verdict is PASS, 1 otherwise, so the
summary SLURM job's own state reflects the run result.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Loguru lines look like "... | ERROR -> message"; match the level field
# rather than the free-text message to avoid false positives.
_LOGURU_ERROR_RE = re.compile(r"\|\s*(ERROR|CRITICAL)\s*->")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")

# sacct states that mean the job did not finish cleanly.
_BAD_STATES = {
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _read_manifest(run_dir: Path):
    """Parse jobs.tsv into a list of dicts. Returns [] if it is missing."""
    path = run_dir / "jobs.tsv"
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        kind, jobid, label, directory = parts[:4]
        rows.append(
            {
                "kind": kind,
                "jobid": jobid,
                "label": label,
                "dir": directory,
            }
        )
    return rows


def _sacct_states(jobids):
    """Return {jobid: {state, exitcode, elapsed}} from a single sacct call.

    Falls back to an empty dict if sacct is unavailable (e.g. when running
    the summary off-cluster), in which case verdicts rely on JUnit/logs only.
    """
    jobids = [j for j in jobids if j and j != "-"]
    if not jobids:
        return {}
    cmd = [
        "sacct",
        "-j",
        ",".join(jobids),
        "--format=JobID,State,ExitCode,Elapsed",
        "--parsable2",
        "--noheader",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}

    states = {}
    for line in out.splitlines():
        fields = line.split("|")
        if len(fields) < 4:
            continue
        jobid, state, exitcode, elapsed = fields[:4]
        # Keep only the top-level job row (skip "<id>.batch", "<id>.extern").
        if "." in jobid:
            continue
        # sacct sometimes annotates, e.g. "CANCELLED by 1234".
        state = state.split()[0] if state else state
        states[jobid] = {
            "state": state,
            "exitcode": exitcode,
            "elapsed": elapsed,
        }
    return states


def _parse_junit(xml_path: Path):
    """Return counts + failed-test ids from a JUnit XML file.

    Handles both a top-level <testsuites> and a bare <testsuite>.
    """
    result = {
        "found": False,
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "failed_ids": [],
    }
    if not xml_path.is_file():
        return result
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return result

    suites = (
        [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    )
    result["found"] = True
    for suite in suites:
        result["tests"] += int(suite.get("tests", 0) or 0)
        result["failures"] += int(suite.get("failures", 0) or 0)
        result["errors"] += int(suite.get("errors", 0) or 0)
        result["skipped"] += int(suite.get("skipped", 0) or 0)
        for case in suite.iter("testcase"):
            if (
                case.find("failure") is not None
                or case.find("error") is not None
            ):
                classname = case.get("classname", "")
                name = case.get("name", "")
                result["failed_ids"].append(
                    f"{classname}::{name}" if classname else name
                )
    result["passed"] = (
        result["tests"]
        - result["failures"]
        - result["errors"]
        - result["skipped"]
    )
    return result


def _scan_logs(directory: str, jobid: str):
    """Scan <directory>/logs/*<jobid>* for error signals.

    Returns (n_error_lines, log_files). Used to decide whether a template
    workflow ran through cleanly even when its job exited 0.
    """
    if not directory or directory == "-":
        return 0, []
    pattern = os.path.join(directory, "logs", f"*{jobid}*")
    files = sorted(glob.glob(pattern))
    n_errors = 0
    for fp in files:
        try:
            with open(fp, "r", errors="replace") as fh:
                for line in fh:
                    if _LOGURU_ERROR_RE.search(line) or _TRACEBACK_RE.search(
                        line
                    ):
                        n_errors += 1
        except OSError:
            continue
    return n_errors, files


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------


def _evaluate(run_dir: Path):
    manifest = _read_manifest(run_dir)
    states = _sacct_states([r["jobid"] for r in manifest])

    tiers = []
    templates = []
    for row in manifest:
        jobid = row["jobid"]
        sacct = states.get(jobid, {})
        state = sacct.get("state", "UNKNOWN")
        elapsed = sacct.get("elapsed", "")

        if row["kind"] == "tier":
            xml = run_dir / f"{row['label']}_{jobid}.xml"
            junit = _parse_junit(xml)
            # A tier passes when JUnit reports no failures/errors. If the XML
            # is missing we fall back to the sacct state.
            if junit["found"]:
                ok = (
                    junit["failures"] == 0
                    and junit["errors"] == 0
                    and state not in _BAD_STATES
                )
            else:
                ok = state == "COMPLETED"
            tiers.append(
                {
                    **row,
                    "state": state,
                    "elapsed": elapsed,
                    "junit": junit,
                    "ok": ok,
                }
            )
        elif row["kind"] == "template":
            n_errors, log_files = _scan_logs(row["dir"], jobid)
            state_bad = state in _BAD_STATES
            # afterany fires only once deps are terminal, so a non-terminal
            # state here means the job never started/finished as expected.
            state_pending = state in {
                "RUNNING",
                "PENDING",
                "REQUEUED",
                "RESIZING",
                "",
            }
            # When sacct is unavailable the state is UNKNOWN; fall back to the
            # log scan alone rather than failing every template.
            ran_through = not state_bad and not state_pending and n_errors == 0
            reasons = []
            if state_bad:
                reasons.append(f"job state {state}")
            elif state_pending:
                reasons.append(f"job not finished (state {state})")
            if n_errors:
                reasons.append(f"{n_errors} error line(s) in logs")
            templates.append(
                {
                    **row,
                    "state": state,
                    "elapsed": elapsed,
                    "n_errors": n_errors,
                    "log_files": log_files,
                    "ran_through": ran_through,
                    "reasons": reasons,
                }
            )

    overall_ok = all(t["ok"] for t in tiers) and all(
        t["ran_through"] for t in templates
    )
    return {
        "tiers": tiers,
        "templates": templates,
        "overall_ok": overall_ok,
        "sacct_available": bool(states),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _read_run_info(run_dir: Path):
    info = {}
    path = run_dir / "run_info.txt"
    if path.is_file():
        for line in path.read_text().splitlines():
            if ":" in line and not line.startswith(" "):
                key, _, val = line.partition(":")
                info[key.strip()] = val.strip()
    return info


def _render_text(run_dir: Path, data: dict) -> str:
    info = _read_run_info(run_dir)
    tiers = data["tiers"]
    templates = data["templates"]
    n_through = sum(t["ran_through"] for t in templates)
    n_templates = len(templates)
    n_tier_fail = sum(not t["ok"] for t in tiers)

    bar = "=" * 70
    sub = "-" * 70
    lines = []
    lines.append(bar)
    lines.append(" picasso-workflow cluster test summary")
    lines.append(bar)
    lines.append(f"run_id:     {info.get('run_id', run_dir.name)}")
    lines.append(
        f"generated:  {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if info.get("git_branch"):
        lines.append(
            f"git:        {info.get('git_branch')} @ {info.get('git_sha', '')}"
        )
    if info.get("project_dir"):
        lines.append(f"project:    {info['project_dir']}")
    if not data["sacct_available"]:
        lines.append(
            "note:       sacct unavailable — job states shown as UNKNOWN; "
            "verdicts use JUnit/logs only"
        )
    lines.append("")

    verdict = "PASS" if data["overall_ok"] else "FAIL"
    headline = (
        f"OVERALL: {verdict}   "
        f"({len(tiers) - n_tier_fail}/{len(tiers)} tiers passed, "
        f"{n_through}/{n_templates} template workflows ran through)"
    )
    lines.append(headline)
    lines.append("")

    # --- Pytest tiers ------------------------------------------------------
    lines.append(sub)
    lines.append(" Pytest tiers")
    lines.append(sub)
    if tiers:
        lines.append(
            f"{'TIER':<10} {'JOB':<9} {'STATE':<11} "
            f"{'TESTS':>5} {'PASS':>5} {'FAIL':>5} {'SKIP':>5}  RESULT"
        )
        for t in tiers:
            j = t["junit"]
            if j["found"]:
                tests = j["tests"]
                passed = j.get("passed", 0)
                failed = j["failures"] + j["errors"]
                skipped = j["skipped"]
            else:
                tests = passed = failed = skipped = "-"
            res = "PASS" if t["ok"] else "FAIL"
            if (
                t["ok"]
                and j["found"]
                and j["tests"] > 0
                and j["skipped"] == j["tests"]
            ):
                res = "PASS (all skipped)"
            lines.append(
                f"{t['label']:<10} {t['jobid']:<9} {t['state']:<11} "
                f"{str(tests):>5} {str(passed):>5} {str(failed):>5} "
                f"{str(skipped):>5}  {res}"
            )
        for t in tiers:
            if t["junit"]["failed_ids"]:
                lines.append("")
                lines.append(f"  {t['label']} failures:")
                for fid in t["junit"]["failed_ids"]:
                    lines.append(f"    - {fid}")
    else:
        lines.append("  (no tier jobs recorded in jobs.tsv)")
    lines.append("")

    # --- Template workflows ------------------------------------------------
    lines.append(sub)
    lines.append(" Template workflows (end-to-end runs)")
    lines.append(sub)
    lines.append(f"Ran through: {n_through} / {n_templates}")
    lines.append("")
    if templates:
        lines.append(
            f"{'NAME':<40} {'JOB':<9} {'STATE':<11} {'ERR':>4}  RESULT"
        )
        for t in templates:
            res = "PASS" if t["ran_through"] else "FAIL"
            name = t["label"]
            if len(name) > 40:
                name = name[:37] + "..."
            lines.append(
                f"{name:<40} {t['jobid']:<9} {t['state']:<11} "
                f"{t['n_errors']:>4}  {res}"
            )
        for t in templates:
            if not t["ran_through"]:
                lines.append("")
                lines.append(
                    f"  {t['label']}: {', '.join(t['reasons']) or 'failed'}"
                )
                for fp in t["log_files"][:4]:
                    lines.append(f"    log: {fp}")
                if not t["log_files"]:
                    lines.append(
                        f"    (no logs found under {t['dir']}/logs/"
                        f"*{t['jobid']}*)"
                    )
    else:
        lines.append(
            "  (no template workflows submitted — PW_TEST_DATA_DIR unset or "
            "no run_workflow_slurm.sh found)"
        )
    lines.append("")
    lines.append(bar)
    lines.append(f" Full artefacts: {run_dir}")
    lines.append(bar)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: summarize.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[0]).resolve()
    if not run_dir.is_dir():
        print(f"run_dir not found: {run_dir}", file=sys.stderr)
        return 2

    data = _evaluate(run_dir)
    text = _render_text(run_dir, data)

    (run_dir / "SUMMARY.txt").write_text(text)
    (run_dir / "summary.json").write_text(json.dumps(data, indent=2) + "\n")

    # Echo to stdout so the report also lands in the summary job's SLURM log.
    print(text)
    return 0 if data["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
