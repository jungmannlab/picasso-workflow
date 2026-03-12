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

# -- Icon (convert .ico -> .icns if sips + iconutil are available) --
ICON_ICO="$(cd "$(dirname "$0")/.." && pwd)/picasso_workflow/picasso-workflow.ico"
ICON_ICNS="$APP_BUNDLE/Contents/Resources/${APP_NAME}.icns"

if [[ -f "$ICON_ICO" ]] && command -v sips &>/dev/null && command -v iconutil &>/dev/null; then
    ICON_TMP="$(mktemp -d)/iconbuild.iconset"
    mkdir -p "$ICON_TMP"
    for SIZE in 16 32 64 128 256 512; do
        sips -z $SIZE $SIZE "$ICON_ICO" \
            --out "$ICON_TMP/icon_${SIZE}x${SIZE}.png" &>/dev/null
        sips -z $((SIZE*2)) $((SIZE*2)) "$ICON_ICO" \
            --out "$ICON_TMP/icon_${SIZE}x${SIZE}@2x.png" &>/dev/null
    done
    iconutil -c icns "$ICON_TMP" -o "$ICON_ICNS"
    echo "Icon: converted picasso-workflow.ico -> icns"
else
    echo "Icon: sips/iconutil not available or .ico missing — no custom icon set."
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
