#!/bin/bash
# Tier 4: real acquired-data tests.
#
# Requires picassosr AND access to a directory of real DNA-PAINT acquisitions
# (OME-TIFF movies from the lab pool volumes).  Tests are skipped
# automatically when PW_TEST_DATA_DIR is not set or the directory is not
# mounted on the compute node.
#
# The pool volumes must be accessible from the compute nodes — confirm with
# your cluster administrator if unsure.
#
# Submit via submit_all.sh (pass PW_TEST_DATA_DIR before calling it) or
# directly:
#   export PW_TEST_DATA_DIR=/path/to/test-datasets
#   sbatch --export=ALL,PW_PROJECT_DIR=/path/to/picasso-workflow \
#          tools/cluster_tests/tier4.sbatch

#SBATCH --job-name=pw-test-tier4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=test-results/tier4_%j.log
#SBATCH --error=test-results/tier4_%j.log

set -uo pipefail

cd "${PW_PROJECT_DIR:?PW_PROJECT_DIR must be set — use submit_all.sh or pass --export=ALL,PW_PROJECT_DIR=...}"
mkdir -p test-results

echo "Job started at: $(date)"
echo "Running on:     $(hostname)"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Project dir:    $(pwd)"
echo "PW_TEST_DATA_DIR: ${PW_TEST_DATA_DIR:-<not set — real_data tests will be skipped>}"
echo ""

module load anaconda/3/2023.03
source ~/.bashrc
conda activate picasso-workflow

python -m pytest \
    -m "integration and real_data" \
    --junit-xml="test-results/tier4_${SLURM_JOB_ID}.xml" \
    -v

echo ""
echo "Job completed at: $(date)"
