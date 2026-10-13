# picasso-workflow one-click installers

This directory builds the friendly, no-Python-required installers for
**Windows** and **macOS**, so lab members with little coding exposure can
install picasso-workflow by double-clicking — and update by downloading the
next release.

## How it works

The installer is built with [`constructor`](https://github.com/conda/constructor)
(BSD-licensed) against the **conda-forge** channel only — the same setup
[Miniforge](https://github.com/conda-forge/miniforge) uses.

> **Licensing:** this deliberately avoids Anaconda's commercial `defaults`
> channel (`repo.anaconda.com`) and the Anaconda/Miniconda installers, both of
> which carry Anaconda's Terms of Service for large organizations. `constructor`
> + conda-forge carry **no** such ToS, so the installer is free to use and
> redistribute at MPG regardless of headcount. Keep it that way: never add a
> channel other than `conda-forge`.

The installer lays down a self-contained conda-forge Python, then
`post_install.{sh,bat}` pip-installs the bundled `picasso_workflow` wheel —
which pulls `picassosr`, PyQt6, PyQt6-WebEngine, numba and the rest from PyPI
wheels (the same pins that `pip install -e .` already uses).

- `construct.yaml` — the recipe (channel, base env, per-user install prefix,
  bundled wheel, branding).
- `post_install.sh` / `post_install.bat` — pip-install the app + create a
  launcher / Start Menu + Desktop shortcuts. Both the install and the launchers
  set `PYTHONNOUSERSITE=1` so a stray user-site directory
  (`~/.local/lib/python3.10` / `%APPDATA%\Python`) can never shadow the bundled
  env — the recurring `~/.local` trap in this stack. On Windows the shortcut
  runs a small `.vbs` launcher (a `.lnk` can't set an env var) that sets the
  flag and starts the GUI with no console flash.
- `wheels/` — the `picasso_workflow` wheel is staged here at build time
  (git-ignored; CI builds it).

### How it launches

- **Windows:** per-user Start Menu + Desktop shortcuts (created by
  `post_install.bat`).
- **macOS:** `post_install.sh` builds a `picasso-workflow.app` in
  `~/Applications`, so it launches from **Spotlight (⌘-Space)** and the
  Applications folder like a normal Mac app. The bundle is created on-device
  (not downloaded), so Gatekeeper does not block it. A
  `Launch picasso-workflow.command` in the install folder is kept as a fallback.

### Per-user, multi-user (Windows)

The recipe installs **per user** under `%LOCALAPPDATA%\picasso-workflow` with no
admin prompt, writing only to the user profile and `HKCU`. Several users on the
same machine can install and run simultaneously without colliding — unlike
picasso's installer, which installs system-wide to `C:\Picasso`.

## Distribution

Installers are built automatically by `.github/workflows/release.yml` on every
`vX.Y.Z` tag push and attached to that tag's **GitHub Release**. Three are
produced:

- **Windows** (`…-Windows-x86_64.exe`)
- **macOS, Apple Silicon** (`…-MacOSX-arm64.pkg`) — M1/M2/M3/M4 Macs
- **macOS, Intel** (`…-MacOSX-x86_64.pkg`) — older Intel Macs

`constructor` builds per host architecture (conda packages are arch-specific;
a universal2 conda env isn't practical), so the two macOS installers are built
on separate Intel (`macos-13`) and Apple-Silicon (`macos-14`) runners. Users
pick the one matching their Mac (About This Mac → "Chip"/"Processor").

The installers are **unsigned** for now, so first launch shows a one-time
Windows SmartScreen ("More info → Run anyway") or macOS Gatekeeper warning.
Code-signing/notarization can be added later. A click-by-click guide for the
macOS first-open step (and how a lab admin can distribute without any prompt)
is in **[`MACOS_FIRST_OPEN.md`](MACOS_FIRST_OPEN.md)**.

## Build locally (optional; the authoritative build is CI)

Requires a conda-forge-based conda (Miniforge recommended) on the target OS —
`constructor` produces an installer **for the OS it runs on**, so build Windows
installers on Windows and macOS installers on macOS.

```bash
# from a Miniforge prompt, in this repo
conda install -c conda-forge constructor
python -m build --wheel --outdir release/wheels .     # stage the app wheel
export PW_VERSION=$(python -m setuptools_scm)          # or e.g. 0.4.0
# fill the version + @WHEEL@ placeholders in construct.yaml (CI does this too)
python - <<'PY'
import glob, os, re, pathlib
wheel = os.path.basename(sorted(glob.glob("release/wheels/*.whl"))[-1])
p = pathlib.Path("release/construct.yaml"); t = p.read_text()
t = re.sub(r"^version:.*$", "version: " + os.environ["PW_VERSION"], t, 1, re.M)
p.write_text(t.replace("@WHEEL@", wheel))
PY
cd release
constructor . --output-dir dist
```

The finished installer lands in `release/dist/`.

> **What CI does and does not prove.** A green CI run means the **installer was
> produced**. The `post_install` pip step (which pulls PyQt6-WebEngine / numba /
> picassosr) only runs when an end user actually **installs** on their machine —
> it is baked into the installer, not executed during the build. So the real
> end-to-end proof is running the produced installer on a clean Windows / macOS
> box; the likeliest spot to need a tweak there is that `pip install`.
