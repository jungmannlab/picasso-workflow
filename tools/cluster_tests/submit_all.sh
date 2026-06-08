#!/bin/bash
# Submit all four test tiers to SLURM with afterok dependencies, so each
# tier only starts if the previous one passed.
#
# Usage (run from anywhere on the cluster login node):
#
#   # Tiers 1-3 only (no real data):
#   tools/cluster_tests/submit_all.sh
#
#   # All four tiers including real-data tests:
#   export PW_TEST_DATA_DIR=/path/to/real/datasets
#   tools/cluster_tests/submit_all.sh
#
# The project directory is inferred automatically from the location of this
# script, so the script can be called from any directory.
#
# All artefacts of a single invocation are gathered in one per-run
# directory:
#
#   <project>/test-results/<timestamp>_<branch>_<sha>/
#       run_info.txt                       run metadata
#       tier<N>_<jobid>.log                SLURM stdout+stderr per tier
#       tier<N>_<jobid>.xml                JUnit XML report per tier
#       picasso-workflow-job<jobid>-rank<rank>.log
#                                          the picasso-workflow loguru log
#
# A <project>/test-results/latest symlink always points at the most recent
# run directory, so downstream analysis can just read test-results/latest/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# One directory per run so artefacts from different runs never intermingle.
# Tag it with the git branch and short SHA when available for traceability.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
GIT_BRANCH="$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo nogit)"
GIT_SHA="$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
# Sanitise the branch name (slashes etc.) so it is safe as a path component.
GIT_BRANCH_SAFE="$(printf '%s' "$GIT_BRANCH" | tr '/ ' '__')"
RUN_ID="${RUN_TS}_${GIT_BRANCH_SAFE}_${GIT_SHA}"
RUN_DIR="$PROJECT_DIR/test-results/$RUN_ID"

mkdir -p "$RUN_DIR"

# Refresh the convenience symlink to the newest run directory.
ln -sfn "$RUN_ID" "$PROJECT_DIR/test-results/latest"

echo "Project directory: $PROJECT_DIR"
echo "Results directory: $RUN_DIR"
echo "                   (also reachable via test-results/latest)"

# Resolve PW_TEST_DATA_DIR: env var takes priority, then config.yaml.
# This mirrors the lookup order in the network_test_data pytest fixture.
if [[ -z "${PW_TEST_DATA_DIR:-}" ]]; then
    _cfg_dir=$(python3 - <<'EOF' 2>/dev/null || true
import pathlib, sys
try:
    import yaml
except ImportError:
    sys.exit(0)
cfg = pathlib.Path.home() / ".config" / "picasso_workflow" / "config.yaml"
if cfg.exists():
    data = yaml.safe_load(cfg.read_text()) or {}
    val = (data.get("TestData") or {}).get("directory")
    if val:
        print(val, end="")
EOF
)
    if [[ -n "${_cfg_dir:-}" ]]; then
        export PW_TEST_DATA_DIR="${_cfg_dir}"
    fi
fi

if [[ -n "${PW_TEST_DATA_DIR:-}" ]]; then
    echo "Test data:         $PW_TEST_DATA_DIR"
else
    echo "Test data:         not configured (PW_TEST_DATA_DIR unset, no TestData.directory in config.yaml)"
    echo "                   → Tier 4 real_data tests will be skipped inside the job"
fi
echo ""

# Common sbatch flags passed to every job. PW_RUN_DIR tells each sbatch
# script where to write its JUnit XML; PW_LOG_DIR redirects the
# picasso-workflow loguru log into the same per-run directory; and the
# per-tier --output/--error flags (added below) place the SLURM logs there
# too. All three therefore land together in $RUN_DIR.
COMMON=(
    --parsable
    --chdir="$PROJECT_DIR"
    --export="ALL,PW_PROJECT_DIR=$PROJECT_DIR,PW_RUN_DIR=$RUN_DIR,PW_LOG_DIR=$RUN_DIR"
)

# ---------------------------------------------------------------------------
# Tier 1 + 2: unit tests + template validation (no picasso required)
# ---------------------------------------------------------------------------
JID1=$(sbatch "${COMMON[@]}" \
    --output="$RUN_DIR/tier1_2_%j.log" \
    --error="$RUN_DIR/tier1_2_%j.log" \
    "$SCRIPT_DIR/tier1_2.sbatch")
echo "Submitted Tier 1+2 (unit + template):  job $JID1"

# ---------------------------------------------------------------------------
# Tier 3: integration tests with synthetic/bundled data
# Starts only when Tier 1+2 passes (exit 0).
# ---------------------------------------------------------------------------
JID2=$(sbatch "${COMMON[@]}" \
    --dependency=afterok:"$JID1" \
    --output="$RUN_DIR/tier3_%j.log" \
    --error="$RUN_DIR/tier3_%j.log" \
    "$SCRIPT_DIR/tier3.sbatch")
echo "Submitted Tier 3  (integration):        job $JID2  (depends on $JID1)"

# ---------------------------------------------------------------------------
# Tier 4: real acquired-data tests
# Starts only when Tier 3 passes. Tests skip gracefully if PW_TEST_DATA_DIR
# is not set or the directory is not mounted on the compute node.
# ---------------------------------------------------------------------------
JID3=$(sbatch "${COMMON[@]}" \
    --dependency=afterok:"$JID2" \
    --output="$RUN_DIR/tier4_%j.log" \
    --error="$RUN_DIR/tier4_%j.log" \
    "$SCRIPT_DIR/tier4.sbatch")
echo "Submitted Tier 4  (real data):          job $JID3  (depends on $JID2)"

# ---------------------------------------------------------------------------
# Template workflow runs: submit each discovered run_workflow_slurm.sh as a
# real end-to-end workflow job. These are self-contained SBATCH scripts (one
# per template folder) that cd into their folder and run the production
# workflow via start_workflow.py; their stdout/stderr and the picasso-workflow
# log land in <template>/logs/ (per the script's own #SBATCH --output and the
# cwd/logs anchoring in config_logger).
#
# A template is opted in simply by containing a run_workflow_slurm.sh under
# PW_TEST_DATA_DIR. Gated on Tier 3 passing, like Tier 4.
# ---------------------------------------------------------------------------
TEMPLATE_JOBS=()
if [[ -n "${PW_TEST_DATA_DIR:-}" && -d "${PW_TEST_DATA_DIR:-}" ]]; then
    while IFS= read -r _script; do
        [[ -z "$_script" ]] && continue
        _tdir="$(cd "$(dirname "$_script")" && pwd)"
        _jid=$(sbatch --parsable --dependency=afterok:"$JID2" "$_script")
        TEMPLATE_JOBS+=("${_jid}|${_tdir}")
        echo "Submitted template run:                 job $_jid  ($_tdir)"
    done < <(
        find "$PW_TEST_DATA_DIR" -type f -name run_workflow_slurm.sh | sort
    )
fi
if [[ ${#TEMPLATE_JOBS[@]} -eq 0 ]]; then
    echo "Template runs:                          none found under PW_TEST_DATA_DIR"
fi

# Write a machine-readable manifest (one row per job) that summarize.py reads
# to build the final report. Columns: kind <TAB> jobid <TAB> label <TAB> dir.
# For tiers, dir is "-"; for templates it is the template folder (whose
# logs/ holds the per-job logs scanned for the "ran through" verdict).
{
    printf 'tier\t%s\ttier1_2\t-\n' "$JID1"
    printf 'tier\t%s\ttier3\t-\n' "$JID2"
    printf 'tier\t%s\ttier4\t-\n' "$JID3"
    for _tj in ${TEMPLATE_JOBS[@]+"${TEMPLATE_JOBS[@]}"}; do
        printf 'template\t%s\t%s\t%s\n' \
            "${_tj%%|*}" "$(basename "${_tj#*|}")" "${_tj#*|}"
    done
} > "$RUN_DIR/jobs.tsv"

# ---------------------------------------------------------------------------
# Summary job: runs after every tier and template job reaches a terminal
# state (afterany, so it runs whether they passed or failed) and condenses
# all artefacts into one SUMMARY.txt. This is what makes the whole run a
# "submit once, read one file" affair.
# ---------------------------------------------------------------------------
DEP_LIST="afterany:$JID1:$JID2:$JID3"
for _tj in ${TEMPLATE_JOBS[@]+"${TEMPLATE_JOBS[@]}"}; do
    DEP_LIST+=":${_tj%%|*}"
done
JIDS=$(sbatch "${COMMON[@]}" \
    --dependency="$DEP_LIST" \
    --output="$RUN_DIR/summary_%j.log" \
    --error="$RUN_DIR/summary_%j.log" \
    "$SCRIPT_DIR/summary.sbatch")
echo "Submitted summary (report):             job $JIDS  (depends on all above)"

# Write a human-readable manifest so the run directory is self-describing.
{
    echo "run_id:        $RUN_ID"
    echo "submitted_at:  $(date)"
    echo "submitted_by:  $(whoami)@$(hostname)"
    echo "project_dir:   $PROJECT_DIR"
    echo "git_branch:    $GIT_BRANCH"
    echo "git_sha:       $GIT_SHA"
    echo "test_data_dir: ${PW_TEST_DATA_DIR:-<not configured — tier 4 real_data skipped>}"
    echo "tier1_2_job:   $JID1"
    echo "tier3_job:     $JID2"
    echo "tier4_job:     $JID3"
    echo "summary_job:   $JIDS"
    if [[ ${#TEMPLATE_JOBS[@]} -gt 0 ]]; then
        echo "template_jobs:"
        for _tj in "${TEMPLATE_JOBS[@]}"; do
            echo "  ${_tj%%|*} -> ${_tj#*|}  (logs in <template>/logs/)"
        done
    fi
} > "$RUN_DIR/run_info.txt"

# Assemble the full job-id list (tiers + template runs + summary) for
# monitoring.
ALL_JIDS="$JID1,$JID2,$JID3"
for _tj in ${TEMPLATE_JOBS[@]+"${TEMPLATE_JOBS[@]}"}; do
    ALL_JIDS+=",${_tj%%|*}"
done
ALL_JIDS+=",$JIDS"

echo ""
echo "Monitor:  squeue -j $ALL_JIDS"
echo "Tail log: tail -f $RUN_DIR/tier1_2_${JID1}.log"
echo "Report:   $RUN_DIR/SUMMARY.txt  (written when summary job $JIDS finishes)"
echo "          → also reachable at test-results/latest/SUMMARY.txt"
echo "Results:  $RUN_DIR  (or test-results/latest)"
echo ""
echo "On-demand snapshot before the run finishes:"
echo "  python3 tools/cluster_tests/summarize.py $RUN_DIR"
