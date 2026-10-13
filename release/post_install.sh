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

# macOS: also install a .app into ~/Applications so the GUI is launchable from
# Spotlight (Cmd+Space) and the Applications folder, like a normal Mac app.
# Per-user (~/Applications needs no admin) and Spotlight-indexed. The bundle is
# created on-device, so it is not quarantined and Gatekeeper does not block it.
if [ "$(uname)" = "Darwin" ]; then
    APP="$HOME/Applications/picasso-workflow.app"
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

    # Nudge LaunchServices/Spotlight to index it immediately.
    lsreg="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
    [ -x "$lsreg" ] && "$lsreg" -f "$APP" 2>/dev/null || true
    touch "$APP"
fi

exit 0
