#!/usr/bin/env python
"""
snapshot_templates.py
---------------------
Developer utility: copy start_workflow.py from each configured template
directory into tests/TestData/templates/ so it can be version-controlled and
used in automated tests.

Run this manually on a machine that can reach the network volumes whenever a
production template changes:

    python tools/snapshot_templates.py

Only start_workflow.py is copied.  YAML file-lists (src_loc.yaml,
raw_locs_list.yaml) are intentionally skipped because they contain absolute
paths to acquired data that must not enter the repository.

The script reads config.yaml in the following priority order:
  1. User-specific: ~/.config/picasso_workflow/config.yaml  (Linux / macOS)
                    %APPDATA%\\picasso_workflow\\config.yaml (Windows)
  2. Package default: picasso_workflow/config.yaml  (next to this script's
                       parent directory)
"""

import os
import shutil
import sys
import yaml

# ---------------------------------------------------------------------------
# Locate config.yaml
# ---------------------------------------------------------------------------


def _user_config_path():
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", "")
    else:
        base = os.path.expanduser("~/.config")
    return os.path.join(base, "picasso_workflow", "config.yaml")


def _package_config_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "picasso_workflow", "config.yaml")


def _find_config():
    user = _user_config_path()
    if os.path.isfile(user):
        return user
    pkg = _package_config_path()
    if os.path.isfile(pkg):
        return pkg
    raise FileNotFoundError(
        "Could not find config.yaml in user config dir or package dir."
    )


# ---------------------------------------------------------------------------
# Destination directory
# ---------------------------------------------------------------------------


def _dest_dir():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        repo_root, "picasso_workflow", "tests", "TestData", "templates"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    config_path = _find_config()
    print(f"Reading config from: {config_path}\n")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    templates = config.get("Templates", {})
    if not templates:
        print("No Templates section found in config.yaml — nothing to do.")
        return

    dest_root = _dest_dir()
    os.makedirs(dest_root, exist_ok=True)

    ok = skipped = errors = 0

    for name, folder in templates.items():
        # Strip raw-string markers that may appear in YAML values
        if isinstance(folder, str) and (
            folder.startswith('r"') or folder.startswith("r'")
        ):
            folder = folder[2:-1]

        src = os.path.join(folder, "start_workflow.py")
        dest_dir = os.path.join(dest_root, name)

        if not os.path.isfile(src):
            print(f"  SKIP  {name!r:<40} start_workflow.py not found at {src}")
            skipped += 1
            continue

        try:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src, dest_dir)
            print(f"  OK    {name!r:<40} <- {src}")
            ok += 1
        except OSError as exc:
            print(f"  ERROR {name!r:<40} {exc}")
            errors += 1

    print(f"\nDone: {ok} copied, {skipped} skipped, {errors} errors.")
    if ok:
        print(f"\nCommit the new/updated files under:\n  {dest_root}")


if __name__ == "__main__":
    main()
