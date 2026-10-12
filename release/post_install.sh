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
# PyPI wheels.
"$PYTHON" -m pip install --no-input --no-warn-script-location \
    --find-links "$PREFIX" picasso_workflow

# A clickable launcher in the install root. `picasso-workflow-gui` is the
# console-script entry point created by the pip install above.
LAUNCHER="$PREFIX/Launch picasso-workflow.command"
cat > "$LAUNCHER" <<EOF
#!/bin/sh
exec "$PREFIX/bin/picasso-workflow-gui"
EOF
chmod +x "$LAUNCHER"

exit 0
