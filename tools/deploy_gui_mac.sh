#!/usr/bin/env bash
# Creates a picasso-workflow.app bundle in ~/Applications so the GUI can be
# launched from Finder or pinned to the Dock — no terminal needed.
#
# Usage (run from anywhere, no sudo required):
#
#   bash tools/deploy_gui_mac.sh
#
# The script auto-detects the active conda environment.  Override with:
#
#   CONDA_ENV_PATH=/path/to/envs/picasso-workflow bash tools/deploy_gui_mac.sh

set -euo pipefail

APP_NAME="picasso-workflow"
APP_BUNDLE="$HOME/Applications/${APP_NAME}.app"

# ---------------------------------------------------------------------------
# 1. Resolve the conda environment
# ---------------------------------------------------------------------------
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_PREFIX:-}}"

if [[ -z "$CONDA_ENV_PATH" ]]; then
    echo "ERROR: No conda environment active and CONDA_ENV_PATH is not set."
    echo "       Activate the environment first:"
    echo "           conda activate picasso-workflow"
    echo "       or pass the path explicitly:"
    echo "           CONDA_ENV_PATH=~/miniconda3/envs/picasso-workflow bash tools/deploy_gui_mac.sh"
    exit 1
fi

if [[ ! -d "$CONDA_ENV_PATH" ]]; then
    echo "ERROR: Conda environment path not found: $CONDA_ENV_PATH"
    exit 1
fi

EXEC_PATH="$CONDA_ENV_PATH/bin/picasso-workflow-gui"
if [[ ! -f "$EXEC_PATH" ]]; then
    echo "ERROR: Executable not found: $EXEC_PATH"
    echo "       Make sure the package is installed:"
    echo "           pip install -e /path/to/picasso-workflow"
    exit 1
fi

# Derive the conda base dir (two levels up from <base>/envs/<name>)
CONDA_BASE="$(cd "$CONDA_ENV_PATH/../.." && pwd)"
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: Cannot find conda.sh at $CONDA_SH"
    echo "       Is CONDA_ENV_PATH pointing inside a real conda installation?"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Build the .app bundle structure
# ---------------------------------------------------------------------------
mkdir -p "$HOME/Applications"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# -- Launcher script (runs inside the bundle, no shell environment assumed) --
cat > "$APP_BUNDLE/Contents/MacOS/$APP_NAME" <<LAUNCHER
#!/usr/bin/env bash
# Use the user's home directory as working directory so that relative paths
# (e.g. the logs/ directory created by config_logger) land somewhere writable.
cd "\$HOME"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_PATH}"
exec "${EXEC_PATH}" "\$@"
LAUNCHER
chmod +x "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

# -- Info.plist --
cat > "$APP_BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>picasso-workflow</string>
    <key>CFBundleIdentifier</key>
    <string>de.jungmannlab.picasso-workflow</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIconFile</key>
    <string>${APP_NAME}</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
PLIST

# -- Icon: .ico -> .icns via Python/Pillow + iconutil --
# sips cannot reliably read .ico files; Pillow (installed with the package)
# handles them correctly.
ICON_ICO="$(cd "$(dirname "$0")/.." && pwd)/picasso_workflow/picasso-workflow.ico"
ICON_ICNS="$APP_BUNDLE/Contents/Resources/${APP_NAME}.icns"

if [[ -f "$ICON_ICO" ]] && command -v iconutil &>/dev/null; then
    ICON_TMP="$(mktemp -d)/iconbuild.iconset"
    mkdir -p "$ICON_TMP"

    python3 - "$ICON_ICO" "$ICON_TMP" <<'PYEOF'
import sys
from pathlib import Path
from PIL import Image

ico_path = Path(sys.argv[1])
out_dir  = Path(sys.argv[2])
img = Image.open(ico_path).convert("RGBA")

for size in [16, 32, 64, 128, 256, 512]:
    for scale, suffix in [(1, ""), (2, "@2x")]:
        px = size * scale
        resized = img.resize((px, px), Image.LANCZOS)
        resized.save(out_dir / f"icon_{size}x{size}{suffix}.png")
PYEOF

    if iconutil -c icns "$ICON_TMP" -o "$ICON_ICNS" 2>/dev/null; then
        echo "Icon: converted picasso-workflow.ico -> icns"
    else
        echo "Icon: iconutil failed — no custom icon set."
    fi
else
    echo "Icon: .ico missing or iconutil not available — no custom icon set."
fi

# ---------------------------------------------------------------------------
# 3. Tell macOS about the new bundle
# ---------------------------------------------------------------------------
touch "$APP_BUNDLE"

echo ""
echo "App bundle created: $APP_BUNDLE"
echo ""
echo "To add it to the Dock: drag $APP_BUNDLE onto the Dock."
echo "To create a Desktop alias: open ~/Applications and drag the app to ~/Desktop."
