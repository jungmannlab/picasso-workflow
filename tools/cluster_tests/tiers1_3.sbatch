#!/bin/bash
# Tiers 1-3: unit tests, template structural validation, and synthetic-data
# integration tests — all in a SINGLE SLURM allocation.
#
# This replaces the former two chained jobs (tier1_2.sbatch + tier3.sbatch).
# Merging them removes a second queue wait and a second conda-env bootstrap,
# which are the two dominant costs of the cluster-CI wall-clock — the tests
# themselves are only a handful. The unit/template tier still GATES the
# integration tier: if the unit tests fail the integration tier is skipped
# and the job exits non-zero, exactly as the old `--dependency=afterok` did.
#
# Two separate pytest invocations are used (rather than one combined run) so
# the summarizer keeps seeing the same per-tier JUnit XML files
# (tier1_2_<jobid>.xml and tier3_<jobid>.xml). --durations=0 makes the SLURM
# log report per-test wall-time, so the queue-wait-vs-compute split is visible.
#
# Requires picassosr in the conda environment (for the integration tier).
# Uses a session-scoped 2 500-frame synthetic OME-TIFF (written to the node's
# /tmp by pytest) and the minimal bundled datasets in
# picasso_workflow/tests/TestData/integration/. No network volumes required.
#
# Submit via submit_all.sh or directly:
#   sbatch --export=ALL,PW_PROJECT_DIR=/path/to/picasso-workflow \
#          tools/cluster_tests/tiers1_3.sbatch

#SBATCH --job-name=pw-test-tiers1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=test-results/tiers1_3_%j.log
#SBATCH --error=test-results/tiers1_3_%j.log

set -uo pipefail

cd "${PW_PROJECT_DIR:?PW_PROJECT_DIR must be set — use submit_all.sh or pass --export=ALL,PW_PROJECT_DIR=...}"

# When launched via submit_all.sh, PW_RUN_DIR is the shared per-run
# directory that also holds the SLURM .log files and the workflow loguru
# log. Fall back to test-results/ for direct sbatch invocation.
RUN_DIR="${PW_RUN_DIR:-test-results}"
mkdir -p "$RUN_DIR"

echo "Job started at: $(date)"
echo "Running on:     $(hostname)"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Project dir:    $(pwd)"
echo "Results dir:    $RUN_DIR"
echo "CPUs:           ${SLURM_CPUS_PER_TASK:-?}"
echo ""

module load anaconda/3/2023.03
source ~/.bashrc
conda activate picasso-workflow

# Confluence (test) credentials live in a dedicated secrets file rather than
# ~/.bashrc, whose non-interactive guard would `return` before reaching any
# exports inside a batch job. The file has no such guard, so it always runs.
[ -f ~/.picasso_secrets ] && source ~/.picasso_secrets

# Let picasso's parallel localisation fit (fit_parallel=True) and any BLAS
# threads use the whole allocation rather than a single core.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

# ---------------------------------------------------------------------------
# Tier 1 + 2: unit tests + template structural validation (the gate).
# ---------------------------------------------------------------------------
echo "=================================================================="
echo " Tiers 1+2: unit tests + template validation"
echo "=================================================================="
python -m pytest \
    -m "not integration" \
    --junit-xml="$RUN_DIR/tier1_2_${SLURM_JOB_ID}.xml" \
    --durations=0 \
    -v
unit_rc=$?

if [[ $unit_rc -ne 0 ]]; then
    echo ""
    echo "Tiers 1+2 failed (pytest exit $unit_rc) — skipping the integration"
    echo "tier, mirroring the old afterok dependency. Job exits non-zero."
    echo "Job completed at: $(date)"
    exit "$unit_rc"
fi

# ---------------------------------------------------------------------------
# Tier 3: integration tests against synthetic and bundled data.
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo " Tier 3: integration (synthetic + bundled data)"
echo "=================================================================="
python -m pytest \
    -m "integration and not real_data" \
    --junit-xml="$RUN_DIR/tier3_${SLURM_JOB_ID}.xml" \
    --durations=0 \
    -v
int_rc=$?

echo ""
echo "Job completed at: $(date)"
exit "$int_rc"
