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
  launcher / Start Menu + Desktop shortcuts.
- `wheels/` — the `picasso_workflow` wheel is staged here at build time
  (git-ignored; CI builds it).

### Per-user, multi-user (Windows)

The recipe installs **per user** under `%LOCALAPPDATA%\picasso-workflow` with no
admin prompt, writing only to the user profile and `HKCU`. Several users on the
same machine can install and run simultaneously without colliding — unlike
picasso's installer, which installs system-wide to `C:\Picasso`.

## Distribution

Installers are built automatically by `.github/workflows/release.yml` on every
`vX.Y.Z` tag push and attached to that tag's **GitHub Release**. Non-coders just
download the `.exe` (Windows) or `.pkg` (macOS) from the Releases page.

The installers are **unsigned** for now, so first launch shows a one-time
Windows SmartScreen ("More info → Run anyway") or macOS Gatekeeper
(right-click → Open) warning. Code-signing/notarization can be added later.

## Build locally (optional; the authoritative build is CI)

Requires a conda-forge-based conda (Miniforge recommended) on the target OS —
`constructor` produces an installer **for the OS it runs on**, so build Windows
installers on Windows and macOS installers on macOS.

```bash
# from a Miniforge prompt, in this repo
conda install -c conda-forge constructor
python -m build --wheel --outdir release/wheels .     # stage the app wheel
export PW_VERSION=$(python -m setuptools_scm)          # or e.g. 0.4.0
cd release
constructor . --output-dir dist
```

The finished installer lands in `release/dist/`.

> **Note:** these files have not yet been proven on a real Windows/macOS build —
> the first CI run on a tag (or `workflow_dispatch` dry run) is the proving
> ground. The most likely spot to need a tweak is the `pip install` of the
> PyQt6-WebEngine / numba stack in `post_install`.
