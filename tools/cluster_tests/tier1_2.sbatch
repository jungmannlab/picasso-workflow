#!/bin/bash
# Tier 1 + 2: unit tests and template structural validation.
#
# No picasso installation or data files required. These tests run the
# mocked picasso_workflow pipeline and check snapshotted template names
# against the AutoPicasso module registry.
#
# Submit via submit_all.sh or directly:
#   sbatch --export=ALL,PW_PROJECT_DIR=/path/to/picasso-workflow \
#          tools/cluster_tests/tier1_2.sbatch

#SBATCH --job-name=pw-test-tier1-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=test-results/tier1_2_%j.log
#SBATCH --error=test-results/tier1_2_%j.log

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

python -m pytest \
    -m "not integration" \
    --junit-xml="$RUN_DIR/tier1_2_${SLURM_JOB_ID}.xml" \
    -v

echo ""
echo "Job completed at: $(date)"
