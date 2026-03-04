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
# Results land in <project>/test-results/ as both a plain log
# (tier<N>_<jobid>.log) and a JUnit XML report (tier<N>_<jobid>.xml).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$PROJECT_DIR/test-results"

echo "Project directory: $PROJECT_DIR"
echo "Results directory: $PROJECT_DIR/test-results"

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

# Common sbatch flags passed to every job.
COMMON=(
    --parsable
    --chdir="$PROJECT_DIR"
    --export="ALL,PW_PROJECT_DIR=$PROJECT_DIR"
)

# ---------------------------------------------------------------------------
# Tier 1 + 2: unit tests + template validation (no picasso required)
# ---------------------------------------------------------------------------
JID1=$(sbatch "${COMMON[@]}" \
    "$SCRIPT_DIR/tier1_2.sbatch")
echo "Submitted Tier 1+2 (unit + template):  job $JID1"

# ---------------------------------------------------------------------------
# Tier 3: integration tests with synthetic/bundled data
# Starts only when Tier 1+2 passes (exit 0).
# ---------------------------------------------------------------------------
JID2=$(sbatch "${COMMON[@]}" \
    --dependency=afterok:"$JID1" \
    "$SCRIPT_DIR/tier3.sbatch")
echo "Submitted Tier 3  (integration):        job $JID2  (depends on $JID1)"

# ---------------------------------------------------------------------------
# Tier 4: real acquired-data tests
# Starts only when Tier 3 passes. Tests skip gracefully if PW_TEST_DATA_DIR
# is not set or the directory is not mounted on the compute node.
# ---------------------------------------------------------------------------
JID3=$(sbatch "${COMMON[@]}" \
    --dependency=afterok:"$JID2" \
    "$SCRIPT_DIR/tier4.sbatch")
echo "Submitted Tier 4  (real data):          job $JID3  (depends on $JID2)"

echo ""
echo "Monitor:  squeue -j $JID1,$JID2,$JID3"
echo "Tail log: tail -f $PROJECT_DIR/test-results/tier1_2_${JID1}.log"
