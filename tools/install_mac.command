#!/usr/bin/env bash
# Double-click this file in Finder to install picasso-workflow.
#
# What it does:
#   1. Finds your conda installation
#   2. Creates (or updates) the "picasso-workflow" conda environment
#   3. Installs the picasso-workflow package
#   4. Creates ~/Applications/picasso-workflow.app
#
# After installation, launch the GUI from ~/Applications or pin it to the Dock.
#
# NOTE: the first time you double-click this file you may need to allow it
# in System Settings > Privacy & Security > "Open Anyway".

set -euo pipefail

# Change to the project root (parent of the tools/ directory).
# This is necessary because Finder opens .command files with the home
# directory as the working directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "================================================="
echo " picasso-workflow  --  macOS installer"
echo "================================================="
echo ""
echo "Project: $PROJECT_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. Find conda
# ---------------------------------------------------------------------------
CONDA_BASE=""

# Try conda on PATH first
if command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi

# Fall back to common install locations
if [[ -z "$CONDA_BASE" ]]; then
    for DIR in \
        "$HOME/miniconda3" \
        "$HOME/anaconda3" \
        "$HOME/opt/miniconda3" \
        "$HOME/opt/anaconda3" \
        "/opt/miniconda3" \
        "/opt/anaconda3" \
        "/usr/local/miniconda3" \
        "/usr/local/anaconda3"
    do
        if [[ -f "$DIR/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="$DIR"
            break
        fi
    done
fi

if [[ -z "$CONDA_BASE" ]]; then
    echo "ERROR: conda not found."
    echo ""
    echo "Install Miniconda from:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo "Then re-run this script."
    echo ""
    read -rp "Press Enter to close..."
    exit 1
fi

echo "Found conda at: $CONDA_BASE"
echo ""

# Activate conda for this shell session
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 2. Create or update the conda environment
# ---------------------------------------------------------------------------
if conda env list | grep -q "^picasso-workflow "; then
    echo "Conda environment \"picasso-workflow\" already exists."
else
    echo "Creating conda environment \"picasso-workflow\" with Python 3.10 ..."
    conda create -n picasso-workflow python=3.10 -y
fi

conda activate picasso-workflow

# ---------------------------------------------------------------------------
# 3. Install picasso-workflow
# ---------------------------------------------------------------------------
echo ""
echo "Installing picasso-workflow from: $PROJECT_DIR"
pip install -e "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# 4. Build the .app bundle
# ---------------------------------------------------------------------------
echo ""
echo "Creating ~/Applications/picasso-workflow.app ..."
CONDA_ENV_PATH="$CONDA_BASE/envs/picasso-workflow" \
    bash "$SCRIPT_DIR/deploy_gui_mac.sh"

echo ""
echo "================================================="
echo " Done!"
echo ""
echo " Launch the GUI from ~/Applications or drag"
echo " picasso-workflow.app to your Dock."
echo "================================================="
echo ""
read -rp "Press Enter to close..."
