#!/bin/bash
# Summary job: assemble one report for a whole cluster test run.
#
# Submitted automatically by submit_all.sh with an `afterany` dependency on
# every tier and template-workflow job, so it runs once they have all reached
# a terminal state (regardless of pass/fail). It condenses the per-run
# artefacts into <run_dir>/SUMMARY.txt (and summary.json) via summarize.py.
#
# Requires only the Python standard library, so the conda environment is not
# strictly needed; we still activate it for a consistent interpreter.

#SBATCH --job-name=pw-test-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=test-results/summary_%j.log
#SBATCH --error=test-results/summary_%j.log

set -uo pipefail

cd "${PW_PROJECT_DIR:?PW_PROJECT_DIR must be set — use submit_all.sh or pass --export=ALL,PW_PROJECT_DIR=...}"

RUN_DIR="${PW_RUN_DIR:-test-results}"

echo "Summary job started at: $(date)"
echo "Running on:             $(hostname)"
echo "Job ID:                 ${SLURM_JOB_ID}"
echo "Run directory:          $RUN_DIR"
echo ""

module load anaconda/3/2023.03 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
conda activate picasso-workflow 2>/dev/null || true

# summarize.py writes SUMMARY.txt + summary.json and echoes the report here.
# Its exit code (0 = overall PASS, 1 = FAIL) becomes this job's exit code.
python3 tools/cluster_tests/summarize.py "$RUN_DIR"
rc=$?

echo ""
echo "Summary job completed at: $(date)  (overall $([ $rc -eq 0 ] && echo PASS || echo FAIL))"
exit $rc
