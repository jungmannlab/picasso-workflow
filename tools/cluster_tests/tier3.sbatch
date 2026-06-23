#!/bin/bash
# Tier 3: integration tests against synthetic and bundled data.
#
# Requires picassosr to be installed in the conda environment. Uses a
# session-scoped 5 000-frame synthetic OME-TIFF (written to the node's
# /tmp by pytest) and the minimal bundled datasets in
# picasso_workflow/tests/TestData/integration/.  No network volumes are
# required.
#
# Submit via submit_all.sh or directly:
#   sbatch --export=ALL,PW_PROJECT_DIR=/path/to/picasso-workflow \
#          tools/cluster_tests/tier3.sbatch

#SBATCH --job-name=pw-test-tier3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=test-results/tier3_%j.log
#SBATCH --error=test-results/tier3_%j.log

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
echo ""

module load anaconda/3/2023.03
source ~/.bashrc
conda activate picasso-workflow

# Confluence (test) credentials live in a dedicated secrets file rather than
# ~/.bashrc, whose non-interactive guard would `return` before reaching any
# exports inside a batch job. The file has no such guard, so it always runs.
[ -f ~/.picasso_secrets ] && source ~/.picasso_secrets

python -m pytest \
    -m "integration and not real_data" \
    --junit-xml="$RUN_DIR/tier3_${SLURM_JOB_ID}.xml" \
    -v

echo ""
echo "Job completed at: $(date)"
