#!/bin/sh
# macOS/Linux post-install: install picasso_workflow (and its PyPI deps) into
# the freshly-laid-down conda env, then drop a double-clickable launcher.
#
# constructor sets $PREFIX to the install location and runs this with the
# env's own interpreter available at $PREFIX/bin/python.
set -e

PYTHON="$PREFIX/bin/python"

# The bundled wheel is copied flat into $PREFIX (via `extra_files`).
# --find-links lets pip install it while pulling picassosr + the rest from
# PyPI wheels. PYTHONNOUSERSITE=1 keeps a stray ~/.local/lib/python3.10 from
# shadowing the bundled env during the install (the documented ~/.local trap).
PYTHONNOUSERSITE=1 "$PYTHON" -m pip install --no-input \
    --no-warn-script-location --find-links "$PREFIX" picasso_workflow

# A clickable launcher in the install root. `picasso-workflow-gui` is the
# console-script entry point created by the pip install above. The launcher
# exports PYTHONNOUSERSITE=1 so the running GUI never picks up ~/.local
# packages either.
LAUNCHER="$PREFIX/Launch picasso-workflow.command"
cat > "$LAUNCHER" <<EOF
#!/bin/sh
export PYTHONNOUSERSITE=1
exec "$PREFIX/bin/picasso-workflow-gui"
EOF
chmod +x "$LAUNCHER"

# macOS: also install a .app into the user's ~/Applications so the GUI is
# launchable from Spotlight (Cmd+Space) and the Applications folder, like a
# normal Mac app. Spotlight-indexed; created on-device so it is not quarantined
# and Gatekeeper does not block it.
if [ "$(uname)" = "Darwin" ]; then
    # Resolve the REAL user's home. A .pkg postinstall runs as root with
    # HOME=/var/root, so $HOME would put the app in root's home (invisible to
    # the user). Fall back to the console (logged-in) user in that case. The
    # .sh installer runs as the user, so $HOME is already correct there.
    # NB: `set -e` aborts on a bare `VAR=$(cmd)` whose command fails, so append
    # `|| true` to tolerate stat/id/dscl failing (then fall back).
    if [ "$(id -u)" = "0" ]; then
        REAL_USER=$(stat -f%Su /dev/console 2>/dev/null) || REAL_USER=""
        [ -z "$REAL_USER" ] && REAL_USER="${SUDO_USER:-${USER:-root}}"
    else
        REAL_USER=$(id -un 2>/dev/null) || REAL_USER="${USER:-$(whoami)}"
    fi
    REAL_HOME=$(dscl . -read "/Users/$REAL_USER" NFSHomeDirectory 2>/dev/null \
        | awk '{print $2}') || REAL_HOME=""
    [ -z "$REAL_HOME" ] && REAL_HOME="$HOME"

    APP="$REAL_HOME/Applications/picasso-workflow.app"
    rm -rf "$APP"
    mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

    cat > "$APP/Contents/MacOS/picasso-workflow" <<EOF
#!/bin/sh
export PYTHONNOUSERSITE=1
exec "$PREFIX/bin/picasso-workflow-gui"
EOF
    chmod +x "$APP/Contents/MacOS/picasso-workflow"

    # Icon bundled via extra_files (copied flat into the prefix root).
    if [ -f "$PREFIX/picasso-workflow.icns" ]; then
        cp "$PREFIX/picasso-workflow.icns" \
            "$APP/Contents/Resources/picasso-workflow.icns"
    fi

    cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>picasso-workflow</string>
    <key>CFBundleDisplayName</key><string>picasso-workflow</string>
    <key>CFBundleIdentifier</key><string>org.jungmannlab.picasso-workflow</string>
    <key>CFBundleExecutable</key><string>picasso-workflow</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleIconFile</key><string>picasso-workflow.icns</string>
</dict>
</plist>
EOF

    # If we ran as root (.pkg), hand the app to the real user so it is theirs
    # and shows up for them.
    if [ "$(id -u)" = "0" ] && [ "$REAL_USER" != "root" ]; then
        chown -R "$REAL_USER" "$REAL_HOME/Applications/picasso-workflow.app" \
            2>/dev/null || true
    fi

    # Nudge LaunchServices/Spotlight to index it immediately.
    lsreg="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
    [ -x "$lsreg" ] && "$lsreg" -f "$APP" 2>/dev/null || true
    touch "$APP"
fi

exit 0
